============================
Installing sklearnef as root
============================

**Note: Not valid yet!**

    * Requirements? Which numpy version? Which scipy version? Whicht sklearn version?
    * buntu provides: http://packages.ubuntu.com/python-sklearn sklearn version.
    * I require maybe already the 0.16? Maybe a little bit older would also be possible?
    * What about the build-essential? Required?

.. note::

    All installation instructions are for Ubuntu, but they should be simmilar for other distributions.
    
When installed with root privileges, **sklearnef** will be available for all uses of your machine.

To install Python packages from `PyPi <https://pypi.python.org>`_, we recommend `PIP <https://pypi.python.org/pypi/pip>`_

.. code-block:: bash

    sudo apt-get install python-pip

Furthermore, you'll require ``numpy`` and ``scipy``, which you can either install from the repositories

.. code-block:: bash
    
    sudo apt-get install python-numpy python-scipy
    
Or via PIP

.. code-block:: bash

    sudo pip install numpy scipy
    
To enable the graph-cut package, we also need the following   
    
.. code-block:: bash
    
    sudo apt-get install libboost-python-dev build-essential
    
And to enable the loading/saving of various image formats, at least ``nibabel`` and ``pydicom`` should be installed

.. code-block:: bash

    sudo pip install nibabel pydicom
    
Now we can install **MedPy**

.. code-block:: bash

    sudo pip install medpy

