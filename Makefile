PKG_NAME=jpt
PKG_VERSION=$(shell cat version)
PYTHON_PATH=/usr/bin/python3.8
PYTHON_VERSION=$(shell ${PYTHON_PATH} -c "import sys; print('%d.%d' % (sys.version_info.major, sys.version_info.minor))")
ENV_NAME=.venv/${PKG_NAME}-${PKG_VERSION}-py-${PYTHON_VERSION}
BASEDIR=$(shell pwd)
RELEASE_NAME=${PKG_NAME}-${PKG_VERSION}

versioncheck: virtualenv
	test ${PKG_VERSION} = `. ${ENV_NAME}/bin/activate && export PYTHONPATH=${BASEDIR}/src && python -c "import jpt; print(jpt.__version__)"`
	@echo Version check passed: ${PKG_VERSION}

rmvirtualenv:
	@(rm -rf .venv)

virtualenv:
	@(virtualenv ${ENV_NAME})
	@(. ${ENV_NAME}/bin/activate && pip install -r requirements.txt -r requirements-dev.txt)

sdist: virtualenv versioncheck
	@(echo "Build ${PKG_NAME} sdist package...")
	@(. ${ENV_NAME}/bin/activate && python setup.py sdist)

bdist: virtualenv versioncheck
	@(echo "Build ${PKG_NAME} bdist package...")
	@(. ${ENV_NAME}/bin/activate && python setup.py bdist)

wheel: virtualenv versioncheck
	@(echo "Build ${PKG_NAME} bdist_wheel package...")
	@(. ${ENV_NAME}/bin/activate && pip install wheel)
	@(. ${ENV_NAME}/bin/activate && python setup.py bdist_wheel)

release: clean sdist bdist wheel
	@mkdir -p releases/${RELEASE_NAME}
	@cp -r dist/* releases/${RELEASE_NAME}/
	@git tag ${PKG_VERSION}
	@git add releases/${RELEASE_NAME}/
	@git commit releases/${RELEASE_NAME}/ -m 'Added: release ${RELEASE_NAME}.'

all: virtualenv versioncheck tests sdist bdist wheel

tests: virtualenv
	@(echo "Running all tests...")
	@(. ${ENV_NAME}/bin/activate &&\
	export PYTHONPATH=${BASEDIR}/src &&\
	cd test &&\
	python -m unittest discover -s ${BASEDIR}/test -t ${BASEDIR}/test)

clean: rmvirtualenv
	rm -rf dist build src/*.egg-info *.log ~/.pyxbld/
