================================
Multi-Objective Optimization for Expensive Blackbox-Functions
================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

      basics <./description/basics.md>
      demo <>
      api-reference <./api.rst>
      FAQs <./description/faqs.md>
      trouble-shooting <./description/trouble-shooting.md>
      algorithms <./description/moo-algorithms.md>
      sequences <./description/sequences.md>
      reflections <./description/reflections.md>
      paper <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4668407>


Paref is a Python package for algorithmic *problem-tailored*
multi-objective optimization of *expensive blackbox-functions*.
This package contains...

* a series of ready-to-use `MOO algorithms <./description/moo-algorithms.md>`_ for frequently encountered problem types

* a framework to quickly implement and apply fully customized MOO algorithms

* generic and intuitive interfaces for MOO algorithms, black-box functions and more, so solving a MOO problem with Paref requires only minimal effort

The official release is available at PyPi:

.. code-block:: shell

    pip install paref



.. _cards-clickable:

.. card:: Getting Started
    :link: ./notebooks/getting_started.ipynb

    Apply your first MOO algorithm with Paref (2 min read)

.. card:: The Basics
    :link: ./description/basics.md

    Learn the basics of Paref (10 min read)

.. card:: Full Use-Case
    :link: ./notebooks/main_example.ipynb

    Built a fully customized MOO algorithm with Paref (20 min read)
