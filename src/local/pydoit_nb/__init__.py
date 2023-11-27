"""
Running notebooks with :mod:`pydoit`

This package will be split out into a separate repository before merging.

The notes below describe the overview of how things work. These can be put into
docs when we split out pydoit-nb into its own project.

A step is a step in the workflow. It can be composed of one or more notebooks.
Steps are handled by:

- :class:`UnconfiguredNotebookBasedStep` - this describes an unconfigured step
  and provides the structure for injecting a function which can configure the
  steps as well as a method for generating notebook-based doit tasks
    - the configuration function is currently always injected. This is because
      configuration can only be performed at run-time and we want to avoid
      inventing a new set of syntax to describe how to do the run-time joins
      right now (we can create helpers as we go along to make this easier and
      see if we end up with repeating patterns that we want to further abstract)

The notebooks are handled by the following two classes:

- :class:`UnconfiguredNotebook` - just describing a notebook, basically only
  metadata
    - typically statically defined as can be known in advance

- :class:`ConfiguredNotebook` - configured notebook i.e. notebook plus the
  configuration of how we want to run it
    - gets created on the fly as it's only at run-time that we can actually
      know all the configuration etc.
"""
