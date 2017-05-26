# optextens
Software implementation of optical extensometer.
The code can be used to build scripts that compute specimen deformation based on series of photos.
The main purpose of this tool is testing of different methods and their properties.

Requirements
============

* `scipy`
* `numpy`
* `matplotlib`

Usage
=====

An example of the usage of this package is shown in example 01.
Follow these steps to run it:
1. Generate a testing image:
```
python foto_examples/gen_examples.py -m
```
1. Run the example:
```
python examples/ex01_accuracy.py
```

3. See the image and the result of finding the first of the marks in it (the correct position being 35px).
![image](https://github.com/heczis/optextens/blob/master/examples/ex01_image.png)
![results](https://github.com/heczis/optextens/blob/master/examples/ex01_result.png)
