#include <iostream>
#include <jack/jack.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include "ros/ros.h"
#include "std_msgs/ByteMultiArray.h"

#define NFRAMES 1024

jack_port_t* input_port;
jack_port_t* output_portL;
jack_port_t* output_portR;
jack_client_t* client;

ros::Publisher pubAudio;
std_msgs::ByteMultiArray msgAudio;

int jack_callback (jack_nframes_t nframes, void *arg)
{
    jack_default_audio_sample_t *in, *outL, *outR;
    
    in = jack_port_get_buffer (input_port, nframes);
    outL = jack_port_get_buffer (output_portL, nframes);
    outR = jack_port_get_buffer (output_portR, nframes);

    memcpy (outL, in, nframes * sizeof (jack_default_audio_sample_t));
    memcpy (outR, in, nframes * sizeof (jack_default_audio_sample_t));
    memcpy (msgAudio.data.data(), in, nframes * sizeof (jack_default_audio_sample_t));

    pubAudio.publish(msgAudio);
    return 0;
}

void jack_shutdown (void *arg)
{
	exit (1);
}


int main (int argc, char *argv[])
{
    std::cout << "INITIALIZING JACK IN TO OUT NODE ..." << std::endl;
    ros::init(argc, argv, "jack_in_to_out");
    ros::NodeHandle n;
    pubAudio = n.advertise<std_msgs::ByteMultiArray>("/microphone", 100);
    msgAudio.data.resize(NFRAMES * sizeof(jack_default_audio_sample_t));
    ros::Rate loop(100);
    
	const char *client_name = "in_to_out";
	jack_options_t options = JackNoStartServer;
	jack_status_t status;

    client = NULL;
    int attempts=5000;
    while(ros::ok() && client == NULL && --attempts > 0)
    {
        client = jack_client_open (client_name, options, &status);
        loop.sleep();
    }
	if (client == NULL)
    {
        std::cout << "jack_client_open() failed, status = " << status << std::endl;
		if (status & JackServerFailed) 
            std::cout << "Unable to connect to JACK server." << std::endl;
		exit (1);
	}
	
	if (status & JackNameNotUnique)
    {
		client_name = jack_get_client_name(client);
        std::cout << "Warning: Repeated agent name, " << client_name << " has been assigned to us." << std::endl;
	}
	
	jack_set_process_callback (client, jack_callback, 0);
	jack_on_shutdown (client, jack_shutdown, 0);

    std::cout << "Engine sample rate: " <<  jack_get_sample_rate (client) << std::endl;
	//Create ports
	input_port = jack_port_register (client, "input", JACK_DEFAULT_AUDIO_TYPE,JackPortIsInput, 0);
	output_portL = jack_port_register (client, "outputL",JACK_DEFAULT_AUDIO_TYPE,JackPortIsOutput, 0);
	output_portR = jack_port_register (client, "outputR",JACK_DEFAULT_AUDIO_TYPE,JackPortIsOutput, 0);
	if ((input_port == NULL) || (output_portL == NULL) || (output_portR == NULL))
    {
        std::cout << "Could not create agent ports. Have we reached the maximum amount of JACK agent ports?" << std::endl;
		exit (1);
	}
	if (jack_activate (client))
    {
        std::cout << "Cannot activate client." << std::endl;
		exit (1);
	}
	
    std::cout << "Agent activated. (Y)" << std::endl;
    std::cout << "Connecting ports... " << std::endl;
	 
	const char **serverports_names;
	serverports_names = jack_get_ports (client, NULL, NULL, JackPortIsPhysical|JackPortIsOutput);
	if (serverports_names == NULL)
    {
        std::cout << "There is no available physical capture (server output) ports." << std::endl;
		exit (1);
	}
	if (jack_connect (client, serverports_names[0], jack_port_name (input_port)))
    {
        std::cout << "Cannot connect input port." << std::endl;
		exit (1);
	}
	free (serverports_names);
	
	serverports_names = jack_get_ports (client, NULL, NULL, JackPortIsPhysical|JackPortIsInput);
	if (serverports_names == NULL)
    {
        std::cout << "No available physical playback (server input) ports." << std::endl;
		exit (1);
	}
	if (jack_connect (client, jack_port_name (output_portL), serverports_names[0]))
    {
        std::cout << "Cannot connect output ports." << std::endl;
		exit (1);
	}
    if (jack_connect (client, jack_port_name (output_portR), serverports_names[1]))
    {
        std::cout << "Cannot connect output ports." << std::endl;
		exit (1);
	}
	free (serverports_names);
	
	
    std::cout << "I think everything is ok (Y)" << std::endl;
	while(ros::ok())
    {
        ros::spinOnce();
        loop.sleep();
    }
	
	jack_client_close (client);
    std::cout << "Releasing buffer memory..." << std::endl;
    
	exit (0);
}
