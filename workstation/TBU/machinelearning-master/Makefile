#http://blog.jgc.org/2011/07/gnu-make-recursive-wildcard-function.html
rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))

MODULES=$(call rwildcard,core,*.py)

test: check
check:
	python -m doctest $(MODULES)

