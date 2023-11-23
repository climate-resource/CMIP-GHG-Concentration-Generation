"""
Running notebooks with :mod:`pydoit`

This package will be split out into a separate repository before merging.

The notes below describe the overview of how things work. These can be put into
docs when we split out pydoit-nb into its own project.

- :class:`UnconfiguredNotebook` - just describing a notebook, basically only
  metadata
    - typically statically defined as can be known in advance
- :class:`ConfiguredNotebook` - configured notebook i.e. notebook plus the
  configuration of how we want to run it
    - gets created on the fly as it's only at run-time that we can actually
      know all the configuration etc.
    - TODO: add to_doit_task to this class and delete class below
- :class:`NotebookTask` - notebook-based doit task, configured notebook plus
  the targets and dependencies so we can make sensible doit tasks
    - gets created on the fly as it's only at run-time that we can actually
      know all the configuration, output paths etc.


"""
