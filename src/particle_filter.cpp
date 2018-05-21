/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

    //set the number of particles
    num_particles = 500;

    //init all particles to first position (based on estimates of
    //x, y, theta and their uncertainties from GPS) and all weights to 1.
    //add random Gaussian noise to each particle.
    default_random_engine gen;

    //these lines create normal (Gaussian) distributions for x, y and theta
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; i++) {

        Particle particle;

        //set values for each particle
        particle.id = i;
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
        particle.weight = 1.0;

        //add to particles list
        particles.push_back(particle);

        //print the particles to the terminal.
        cout << "Particle  #" << particle.id << " " << particle.x << " " << particle.y;
        cout << " " << particle.theta << " " << particle.weight << endl;

        //init weight lists for random distribution in function resample
        weights.push_back((i));

    }

    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

    //init default_random_engine
    default_random_engine gen;

    //call in main.cpp pf.prediction(delta_t, sigma_pos, previous_velocity, previous_yawrate);
    for (int i = 0; i < num_particles; i++) {

        // predict new values for x, y and theta without noise
        if(yaw_rate > 0.001) {
            particles[i].x += velocity/yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
            particles[i].y += velocity/yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
        }
        else {
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        }

        particles[i].theta += yaw_rate*delta_t;

        //create normal_distribution about origin with corresponding std_deviations
        normal_distribution<double> dist_x(0.0, std_pos[0]);
        normal_distribution<double> dist_y(0.0, std_pos[1]);
        normal_distribution<double> dist_theta(0.0, std_pos[2]);

        //add noise
        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);

        // //print the particles to the terminal.
        // cout << "Particle  #" << particles[i].id << " " << particles[i].x << " " << particles[i].y;
        // cout << " " << particles[i].theta << " " << particles[i].weight << endl;

    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

    //call in main.cpp by pf.updateWeights(sensor_range, sigma_landmark, noisy_observations, map);

    //loop for all particles
    for (int i = 0; i < num_particles; i++) {

        //temporary weight for every single particle
        double tmp_weight = 1.0;

        //loop over all observations
        for(int j = 0; j < observations.size(); j++) {

            //transform into map coordinate space
            std::vector<double> transformed = transform_obs2map(particles[i].x, particles[i].y, particles[i].theta,
                                                                observations[j].x, observations[j].y);

            // cout << "x in mapspace" << transformed[0] << endl;
            // cout << "y in mapspace" << transformed[1] << endl;

            //go through all map landmarks and find nearest neighbor
            //start with first landmark
            int idx_nn = 0;
            double min_dist = dist(transformed[0], \
                                   transformed[1], \
                                   map_landmarks.landmark_list[0].x_f, \
                                   map_landmarks.landmark_list[0].y_f);

            //loop over all other landmarks
            for (int k = 1; k < map_landmarks.landmark_list.size(); k++) {

                // cout << "Lenght landmark_list " << map_landmarks.landmark_list.size() << endl;
                // cout << "ID landmark " << map_landmarks.landmark_list[k].id_i << endl;
                // cout << "x position landmark " << map_landmarks.landmark_list[k].x_f << endl;

                double distance = dist(transformed[0], \
                                       transformed[1], \
                                       map_landmarks.landmark_list[k].x_f, \
                                       map_landmarks.landmark_list[k].y_f);

                if(distance < min_dist) {
                    min_dist = distance;
                    idx_nn = k;
                }
            }

            //calculate intermediate steps for weight update
            double dev_x, dev_y, dev_x2_norm, dev_y2_norm;

            dev_x = (transformed[0] - map_landmarks.landmark_list[idx_nn].x_f);
            dev_x2_norm = dev_x * dev_x / (2 * std_landmark[0] * std_landmark[0]);

            dev_y = (transformed[1] - map_landmarks.landmark_list[idx_nn].y_f);
            dev_y2_norm = dev_y * dev_y / (2 * std_landmark[1] * std_landmark[1]);

            //calculate weight acording to multivariate gaussian
            double gauss_norm = (1 / (2 * M_PI * std_landmark[0] * std_landmark[1]));
            tmp_weight *= gauss_norm * exp(-1.0 * (dev_x2_norm + dev_y2_norm));


        }

        particles[i].weight = tmp_weight;

        // output for debug
        // cout << "Particle  #" << particles[i].id << " " << particles[i].x << " " << particles[i].y;
        // cout << " " << particles[i].theta << " " << particles[i].weight << endl;

    }

}

void ParticleFilter::resample() {

    //init temporary particle set
    std::vector<Particle> tmp_particles;

    //init random distribution to pick index from
    random_device rd;
    mt19937 gen(rd());
    discrete_distribution<> dist_index(weights.begin(), weights.end());

    //variables for index, beta and maximum weight
    int index = dist_index(gen);
    double beta = 0.0;
    double mw = 0.0;

    //get maximum weight
    for (int i = 0; i < num_particles; i++) {
        if(particles[i].weight > mw) {
            mw = particles[i].weight;
        }
    }

    // cout << "Random index " << index << endl;

    //loop for number of particles
    for (int i = 0; i < num_particles; i++) {

        //randomly set beta between 0 and 2 * maximum weight
        beta += double(dist_index(gen)) / double(num_particles - 1) * 2.0 * mw;
        // cout << "Beta " << beta << endl;

        //while beta is bigger than weight of current particle, reduce it by that and count index up by one
        while(beta > particles[index].weight) {
            beta -= particles[index].weight;
            index = (index + 1) % num_particles;
        }
        //else add current particle to temporary list
        tmp_particles.push_back(particles[index]);
    }

    //set temporory particle set as new particles set
    particles = tmp_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
