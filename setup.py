from setuptools import setup#, find_packages

'''with open("README.md", "r") as fh:
    ## skip the HTML, which doesn't work on PyPI
    long_description = "".join(fh.readlines()[4:])'''

setup(
    name='abate',
    version='0.0.dev7',
    author='Everett Schlawin, Keith Baka',
    #packages=['abate'],
    #url="https://github.com/eas342/tshirt",
    description="A package to fit exoplanet atmosphere lightcurves, especially for JWST data",
    include_package_data=True,
    install_requires=[
        "numpy>=1.15",
        "scipy>=1.1.0",
        "astropy>=2.0",
        "tqdm>=4.46.0",
        "photutils>=0.4.1",
        "bokeh>=1.4.0",
        "pytest>=7.1.2",
        "exoplanet>=0.5.2",
        "tshirt>=0.1.dev9",
        "pymc3_ext>=0.1.1",
        "celerite2>=0.2.1",
        "ipython>=8.4.0",
        "starry>=1.2.0",
        "corner>=2.2.1"
    ],
    #long_description=long_description,
    #long_description_content_type='text/markdown'
)
