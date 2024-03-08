:html_theme.sidebar_secondary.remove:

================================
Paref: MOO for Expensive Blackbox-Functions
================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

      basics <./description/basics.md>
      faq <./description/faqs.md>
      trouble-shooting <./description/trouble-shooting.md>
      algorithms <./description/moo-algorithms.md>
      demo <https://huggingface.co/spaces/NicoPalm/paref-showcase>
      api-reference <./api.rst>
      sequences <./description/sequences.md>
      reflections <./description/reflections.md>
      theory <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4668407>


Paref provides *problem-tailored*
`multi-objective optimization <https://en.wikipedia.org/wiki/Multi-objective_optimization>`_ algorithms for *expensive blackbox-functions*
based on the `theory of Pareto reflections <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4668407>`_.
This Python package contains...

* a series of ready-to-use `MOO algorithms <./description/moo-algorithms.md>`_ for frequently encountered problem types

* an info module providing you with the necessary knowledge of your black-box function's Pareto front

* a framework to quickly implement and apply customized MOO algorithms


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
