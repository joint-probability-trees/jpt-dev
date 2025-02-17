PKG_NAME=pyjpt

PKG_VERSION=`test -f src/jpt/.version && cat src/jpt/.version || echo 0.0.0`

ENV_PYTHON_VERSION := $(shell test -n "$(PYTHON_VERSION)" && echo "$(PYTHON_VERSION)" || python --version 2>&1 | awk '{print $$2}')

PY_VERSION_STR=$(shell echo ${ENV_PYTHON_VERSION} | tr -d '.')

ARCH_STR=linux_`uname -p`

PYTHON_CMD=python${ENV_PYTHON_VERSION}

ENV_NAME=.venv/${PKG_NAME}-${PKG_VERSION}-cp${PY_VERSION_STR}

BASEDIR=`pwd`

RELEASE_NAME=${PKG_NAME}-${PKG_VERSION}


preload:
	@(echo Package Name: "${PKG_NAME}")
	@(echo pyJPT Package Version: "${PKG_VERSION}")
	@(echo Python Version: "${ENV_PYTHON_VERSION}")
	@(echo Python Version String: "${PY_VERSION_STR}")
	@(echo Virtual Env Name: "${ENV_NAME}")
	@(echo Architecture: "${ARCH_STR}")

rmvirtualenv: preload
	@(rm -rf .venv)

virtualenv: preload
	@(virtualenv ${ENV_NAME} --python ${PYTHON_CMD})
	@(. ${ENV_NAME}/bin/activate && pip install -U pip)
	@(. ${ENV_NAME}/bin/activate && pip install -U -r requirements.txt) # -r requirements-dev.txt)

sdist: preload virtualenv
	@(echo "Build ${PKG_NAME} sdist package...")
	@(. ${ENV_NAME}/bin/activate && pip install -r requirements-dev.txt && python setup.py sdist)

bdist: preload virtualenv
	@(echo "Build ${PKG_NAME} bdist package...")
	@(. ${ENV_NAME}/bin/activate && pip install -r requirements-dev.txt && python setup.py bdist)

wheel: preload virtualenv
	@(echo "Build ${PKG_NAME} bdist_wheel package...")
	@(. ${ENV_NAME}/bin/activate && pip install -U wheel pip)
	@(. ${ENV_NAME}/bin/activate && pip install -r requirements-dev.txt && python setup.py bdist_wheel)

release: preload clean sdist bdist wheel tests
	@mkdir -p releases/${RELEASE_NAME}
	@cp -r dist/* releases/${RELEASE_NAME}/
	@git tag ${PKG_VERSION}
	@git add releases/${RELEASE_NAME}/
	@git commit releases/${RELEASE_NAME}/ -m 'Added: release ${RELEASE_NAME}.'

all: preload clean virtualenv tests sdist bdist wheel

tests: preload virtualenv wheel
	@(echo "Running all tests...")
	@(. ${ENV_NAME}/bin/activate &&\
	pip install dist/${PKG_NAME}-${PKG_VERSION}-cp${PY_VERSION_STR}-cp${PY_VERSION_STR}-${ARCH_STR}.whl &&\
	cd test &&\
	python -m unittest)

clean: preload rmvirtualenv
	rm -rf dist build src/*.egg-info *.log ~/.pyxbld/

update_pkg: preload
	@(. ${ENV_NAME}/bin/activate && pip install -U pip)
	@(. ${ENV_NAME}/bin/activate && pip install -U `cat requirements.txt`)