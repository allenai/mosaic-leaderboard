FROM library/python:3.6

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY requirements.txt requirements.txt

RUN pip3.6 install sklearn
RUN pip3.6 install -r requirements.txt

COPY . .

RUN mkdir /results

CMD ["/bin/bash"]