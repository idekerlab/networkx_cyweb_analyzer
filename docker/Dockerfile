FROM continuumio/miniconda3


RUN mkdir /tmp/networkxcywebanalyzer
COPY dist/*.whl /tmp/networkxcywebanalyzer/

RUN pip install /tmp/networkxcywebanalyzer/networkxcywebanalyzer*whl

RUN rm -rf /tmp/networkxcywebanalyzer

ENTRYPOINT ["/opt/conda/bin/networkxcywebanalyzercmd.py"]
CMD ["--help"]
