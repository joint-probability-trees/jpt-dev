PKG_NAME=pyjpt

DEFAULT_PYTHON_VERSION=3.11

PKG_VERSION=`git describe --tags --abbrev=0 2>/dev/null | sed 's/^v//' || echo 0.0.0`

ENV_PYTHON_VERSION=`test -n "${PYTHON_VERSION}" && echo ${PYTHON_VERSION} || echo ${DEFAULT_PYTHON_VERSION}`

PY_VERSION_STR=$(shell echo ${ENV_PYTHON_VERSION} | tr -d '.')

PYTHON_CMD=python${ENV_PYTHON_VERSION}

ENV_NAME=.venv/${PKG_NAME}-cp${PY_VERSION_STR}

BASEDIR=`pwd`

preload:
	@(echo Package Name: "${PKG_NAME}")
	@(echo pyJPT Package Version: "${PKG_VERSION}")
	@(echo Python Version: "${ENV_PYTHON_VERSION}")
	@(echo Python Version String: "${PY_VERSION_STR}")
	@(echo Virtual Env Name: "${ENV_NAME}")

rmvirtualenv: preload
	@(rm -rf .venv)

virtualenv: preload
	@(virtualenv ${ENV_NAME} --python ${PYTHON_CMD})
	@(. ${ENV_NAME}/bin/activate && pip install -U pip)
	@(. ${ENV_NAME}/bin/activate && JPT_NO_CYTHON=1 pip install -e ".[dev]")

sdist: preload virtualenv
	@(echo "Build ${PKG_NAME} sdist package...")
	@(. ${ENV_NAME}/bin/activate && python -m build --sdist)

wheel: preload virtualenv
	@(echo "Build ${PKG_NAME} wheel package...")
	@(. ${ENV_NAME}/bin/activate && python -m build --wheel)

dist: preload virtualenv
	@(echo "Build ${PKG_NAME} sdist and wheel packages...")
	@(. ${ENV_NAME}/bin/activate && python -m build)

all: preload clean virtualenv tests dist

tests: preload virtualenv
	@(echo "Running all tests...")
	@(. ${ENV_NAME}/bin/activate &&\
	cd test &&\
	python -m unittest)

clean: preload rmvirtualenv
	rm -rf dist build src/*.egg-info *.log ~/.pyxbld/
	find src/. -type f \( -name "*.so" -o -name "*.cpp" -o -name "*.h" \) -delete

update_pkg: preload
	@(. ${ENV_NAME}/bin/activate && pip install -U pip)
	@(. ${ENV_NAME}/bin/activate && JPT_NO_CYTHON=1 pip install -U -e ".[dev]")
