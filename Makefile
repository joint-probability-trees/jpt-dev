PKG_NAME=pyjpt
PKG_VERSION=$(cat src/jpt/.version || echo 0.0.0)
PYTHON_VERSION=3.10
PY_VERSION_STR=$(awk -F '.' '{print $1 $2}' <<< "${PYTHON_VERSION}")
ARCH_STR=linux_x86_64
PYTHON_PATH=python${PYTHON_VERSION}

ENV_NAME=.venv/${PKG_NAME}-${PKG_VERSION}-cp${PY_VERSION_STR}
BASEDIR=$(shell pwd)
RELEASE_NAME=${PKG_NAME}-${PKG_VERSION}

versioncheck: virtualenv
	test ${PKG_VERSION} = `. ${ENV_NAME}/bin/activate && export PYTHONPATH=${BASEDIR}/src/jpt && python -c "import version; print(version.__version__)"`
	@echo Version check passed: ${PKG_VERSION}

rmvirtualenv:
	@(rm -rf .venv)

virtualenv:
	@(virtualenv ${ENV_NAME} --python ${PYTHON_PATH})
	@(. ${ENV_NAME}/bin/activate && pip install -U pip)
	@(. ${ENV_NAME}/bin/activate && pip install -U -r requirements.txt) # -r requirements-dev.txt)

sdist: virtualenv versioncheck
	@(echo "Build ${PKG_NAME} sdist package...")
	@(. ${ENV_NAME}/bin/activate && pip install -r requirements-dev.txt && python setup.py sdist)

bdist: virtualenv versioncheck
	@(echo "Build ${PKG_NAME} bdist package...")
	@(. ${ENV_NAME}/bin/activate && pip install -r requirements-dev.txt && python setup.py bdist)

wheel: # virtualenv
	@(echo "Build ${PKG_NAME} bdist_wheel package...")
	@(. ${ENV_NAME}/bin/activate && pip install -U wheel pip)
	@(. ${ENV_NAME}/bin/activate && pip install -r requirements-dev.txt && python setup.py bdist_wheel)

release: clean sdist bdist wheel tests
	@mkdir -p releases/${RELEASE_NAME}
	@cp -r dist/* releases/${RELEASE_NAME}/
	@git tag ${PKG_VERSION}
	@git add releases/${RELEASE_NAME}/
	@git commit releases/${RELEASE_NAME}/ -m 'Added: release ${RELEASE_NAME}.'

all: clean virtualenv versioncheck tests sdist bdist wheel

tests: wheel
	@(echo "Running all tests...")
	@(. ${ENV_NAME}/bin/activate &&\
	pip install dist/${PKG_NAME}-${PKG_VERSION}-cp${PY_VERSION_STR}-cp${PY_VERSION_STR}-${ARCH_STR}.whl &&\
	cd test &&\
	python -m unittest)

clean: rmvirtualenv
	rm -rf dist build src/*.egg-info *.log ~/.pyxbld/

update_pkg:
	@(. ${ENV_NAME}/bin/activate && pip install -U pip)
	@(. ${ENV_NAME}/bin/activate && pip install -U `cat requirements.txt`)