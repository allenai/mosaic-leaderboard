FROM library/python:3.6

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /local

RUN pip3.6 install scikit-learn pandas numpy

COPY eval.py eval.py

RUN mkdir /results

CMD ["/bin/bash"]