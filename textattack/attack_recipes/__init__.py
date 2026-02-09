""".. _attack_recipes:

Attack Recipes:
======================

We provide a number of pre-built attack recipes, which correspond to attacks from the literature. To run an attack recipe from the command line, run::

    textattack attack --recipe [recipe_name]

To initialize an attack in Python script, use::

    <recipe name>.build(model_wrapper)

For example, ``attack = InputReductionFeng2018.build(model)`` creates `attack`, an object of type ``Attack`` with the goal function, transformation, constraints, and search method specified in that paper. This object can then be used just like any other attack; for example, by calling ``attack.attack_dataset``.

TextAttack supports the following attack recipes (each recipe's documentation contains a link to the corresponding paper):

.. contents:: :local:
"""

from .attack_recipe import AttackRecipe

from .textfooler_jin_2019 import TextFoolerJin2019
from .pwws_ren_2019 import PWWSRen2019
from .bae_garg_2019 import BAEGarg2019

