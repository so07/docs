# Python testing

## [Travis](https://travis-ci.org/)

example file `.travis.yml`
```
language: python
python:
  - "3.6"
  - "3.7"
  - "3.8"

install:
  - pip install .
  - pip install coverage
  - pip install coveralls

script:
  - pytest

after_success:
  - coverage run -m pytest
  - coverage report
  - coverage html
  - coveralls

notifications:
   email:
      on_success: change
      on_failure: always
```

## [Coveralls](https://coveralls.io/)

example of file `.coveragerc`

```
[run]
    source = SOURCE_DIR

[report]
    exclude_lines =
        pragma: no cover
        def __repr__
        if self.debug:
        if settings.DEBUG
        raise AssertionError
        raise NotImplementedError
        if 0:
        if __name__ == .__main__.:
```

example of file `.coveralls.yml`

```
repo_token: HERE_YOUR_TOKEN
```
