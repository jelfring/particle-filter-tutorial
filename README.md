# Installation
All code is written in Python. In order to run the code, the following packages must be installed:

* numpy 
* matplotlib
* scipy

Installing these packages can be done using:
```sh
$ pip install <packagename>
```

# Running the example
To run a particle filter for robot localization in the simulated world, run the demo script `demo_running_example.py` directly in your IDE or use the command line command below.

In Windows:
```sh
$ python3.7.exe .\demo_running_example.py
```

In Linux:
```sh
$ python3 demo_running_example.py
```

The simulation will run and the visualization below should appear.

![alt text](https://github.com/jelfring/particle_filter_tutorial/blob/master/images/running_example_screenshot.png?raw=true)


The picture shows a top view of a 2D simulated world. Four landmarks can be observed by the robot (blue rectangles). The landmark positions are given in the map and therefore are used to estimate the tru robot position and orientation (red circle). The particles that together represent the posterior distribution are represented by the green dots.
