Creating your first fit
=========================

The first thing you want to do when using ABATE is to create a compatible virtual environment and install the ABATE package to it. To do this, you'll need to run the following three commands in your command line:

.. code-block:: python

   import abate
   from abate import fit_GENERIC
   fit_GENERIC.do_fitting()


NOTE: If you've cloned the GitHub (See *Installation*) or downloaded the python scripts from there, you can do the operation performed by the above code in your preferred way.

Running this may take a while, especially if you have not run anything with PyMC3 before. When it's done, it will have created several folders and files within your working directory. One of those directories should be 'plots/lc_plots/mirage_038_hatp14_p001_A3_phot_no_backgsrc_starry_003_fix_LD', which should contain within it an image that looks like this:

.. image:: images/plot_mapsoln_True_mirage_038_hatp14_p001_A3_phot_no_backgsrc_starry_003_fix_LD.png
   :width: 600

If it doesn't, then something's gone wrong! (Hopefully not though)

Fitting Your Own Lightcurve
---------------------------

What do I do with that .yaml file?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using fit_GENERIC.py as a template, we can get to fitting your own lightcurve data! To get this file, make sure you clone the GitHub with the command found in *Installation*.

First, take a look at that .yaml file that was downloaded to  '/parameters/phot_params/jwst_test_data/sim_mirage_038_hatp14_no_backg_source'. This file contains lots of basic but useful information about the observation, and you can either edit this one to match your data, or create your own.

Looking at fit_GENERIC, we can see that this sequence of folders directly corresponds to the variable "paramPath" at the beginning of the function. Modify this so it matches the location of your own .yaml file.


Inputs into simple_pymc_fits.exo_model
---------------------------------------

descrip
~~~~~~~~

This is just a string, and will be what all the final output files and folders will be named as. Change this to your liking (I personally match this to the name of the .yaml file, though it isn't matched in the example).

t$_0$_lit
~~~~~~~~~

This variable is a tuple, with the first value being time, and the other being the uncertainty on the time. You can find this information on the Caltech exoplanet archive, https://exoplanetarchive.ipac.caltech.edu/, along with information for many other values we will be using. If the time is incorrect, it will not fit your data - see *Error Fixes* for more information on how to resolve this issue. Note that a small error in t$_0$_lit can have a large impact on the fit of your lightcurve.

period_lit
~~~~~~~~~~

This variable is a tuple, with the first value being the orbital period in days, and the second value is the uncertainty on the period. This value can be found on the Caltech exoplanet archive, https://exoplanetarchive.ipac.caltech.edu/.

inc_lit
~~~~~~~

This variable is a tuple, with the first value being inclination of the orbit (with 90º being edge on), and the second value being the uncertainty on the inclination. This value can be found on the Caltech exoplanet archive, https://exoplanetarchive.ipac.caltech.edu/, but note that a slight error in inclination can have a noticeable negative outcome on the fit of your lightcurve - see *Error Fixes* for more information on how to resolve this issue.

a_lit
~~~~~~

This variable is a tuple, where the first value is the semimajor axis divided by $R_*$, and the second value is the uncertainty. This information can be found on the Caltech exoplanet archive, https://exoplanetarchive.ipac.caltech.edu/, where it is formatted to the correct unit for this package.

u_lit
~~~~~~~

This variable is a list, and is the limb darkening parameter. It has two main modes:
1. quadratic, where the list is two values long
2. multiterm, where the list has the same number of values as the value of the variable starry_ld_degree

When either of these modes are used, limb darkening becomes a fixed parameter, and is not sampled by PyMC3. However, if *u_lit* is set to *None*, then it becomes a free parameter that is sampled by PyMC3, though it will take longer to run.

To calculate the values needed for the two main modes, you can visit https://exoctk.stsci.edu/limb_darkening, which allows you to select your target and the filter used to collect the data, choose your type(s) of limb darkening law, and calculate the coefficients needed. For quadratic, you will use the values under *c1* and *c2* in the table, and for multiterm, you will use **DR SCHLAWIN, PLEASE FILL IN!!!**

starry_ld_degree
~~~~~~~~~~~~~~~~~

This variable is an integer, and is the polynomial degree starry will use to fit the star and planet to. The higher this number, the more accurate the results, but the longer the function will take to run. It is set to 6 by default.

ecc
~~~~

This variable is a tuple, with the first value being the eccentricity of the orbit, and the second value being the uncertainty. This value can be found on the Caltech exoplanet archive, https://exoplanetarchive.ipac.caltech.edu/.

omega
~~~~~~~

This variable is a tuple, with the first value being ____________, and the second value is the uncertainty. This value can be found on the Caltech exoplanet archive, https://exoplanetarchive.ipac.caltech.edu/.

recalculateTshirt
~~~~~~~~~~~~~~~~~~~~~

This variable is a Boolean, either True or False, that determines whether or not Tshirt recalculates values if pipeType = 'spec'. It is set to True by default.

pipeType
~~~~~~~~~

This variable is a string, set to either 'phot' for photometry or 'spec' for spectroscopy.

ld_law
~~~~~~~

This variable is a string, set to either 'quadratic' or 'multiterm', which determines the length of the list for u_lit.

TShirt
------

For more information on *recalculateTshirt* and *pipeType*, visit tshirt's documentation at https://tshirt.readthedocs.io/en/latest/index.html. Tshirt is automatically downloaded as part of the ABATE package, so there is no need to follow the install procedure.