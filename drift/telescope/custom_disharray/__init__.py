"""
Implementations of customisable, dish-array transit telescopes with 
support for multi-pointed surveys. Mixin classes for adding this 
functionality to generic :py:class:`drift.core.telescope.TransitTelescope`'s 
are provided.

Mixins
======
- :py:class:`.core.CustomDishArray`
- :py:class:`.core.MultiElevationSurvey`

Concrete Implementations
========================
- :py:class:`.core.PolarisedDishArray`
- :py:class:`.core.PolarisedDishArraySurvey`
- :py:class:`.core.UnpolarisedDishArray`
- :py:class:`.core.UnpolarisedDishArraySurvey`

- :py:class:`.hirax.HIRAX`
- :py:class:`.hirax.HIRAXSurvey`
- :py:class:`.hirax.HIRAXHexTile`
- :py:class:`.hirax.HIRAXHexTileSurvey`

.. autosummary::
    :toctree:

    core
    beams
    hirax
    layouts

Example Usage
=============
Insert example configuration file here.
"""