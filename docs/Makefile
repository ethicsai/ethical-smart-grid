# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# "Hot reload": generate the doc automatically when a modification is detected.
# This command should be called in the directory where this Makefile lives.
preview:
	sphinx-reload --build-dir "$(BUILDDIR)" .

# Build docs for multiple versions (based on Git branches and tags).
# The result is placed in `build/html-mv` (for "multiversion"), with each
# version getting its own sub-directory inside it.
# Additionally, we place a small fake index.html that simply redirects clients
# to the latest stable version found. If no version can be found, the build aborts.
multiversion:
	sphinx-multiversion "$(SOURCEDIR)" "$(BUILDDIR)/html-mv"
	# Find the tag with the highest version (latest "stable")
	latest=$$(ls -1 "$(BUILDDIR)/html-mv" | grep 'v[0-9]' | sort --reverse --version-sort | head -n 1) ;\
	latest=$${latest%/} ;\
	if [ -z "$$latest" ]; then \
	  echo 'ERROR: Cannot find latest version in tags. Aborting...' && exit 1 ;\
	else \
	  echo "Found latest version: $$latest ; creating redirect..."; \
	  sed "s/{{ version }}/$$latest/g" redirect_to_version.html > "$(BUILDDIR)/html-mv/index.html" ;\
	fi

# Also remove the `source/modules` directory, which is built by autosummary.
superclean: clean
	rm -r "$(SOURCEDIR)/modules"

.PHONY: help Makefile preview

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
