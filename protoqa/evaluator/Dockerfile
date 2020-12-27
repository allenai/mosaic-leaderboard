FROM library/python:3.6

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN python -m pip install -U pip

COPY requirements.txt .
RUN pip install -r requirements.txt

ENV PYTHONPATH .

COPY . .

RUN mkdir /results

CMD ["/bin/bash"]