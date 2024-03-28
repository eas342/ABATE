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

Parameters
~~~~~~~~~~
See description of parameters in :ref:`param_descrip`


TShirt
------

For more information on *recalculateTshirt* and *pipeType*, visit tshirt's documentation at https://tshirt.readthedocs.io/en/latest/index.html. Tshirt is automatically downloaded as part of the ABATE package, so there is no need to follow the install procedure.