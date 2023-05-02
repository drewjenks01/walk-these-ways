# syntax=docker/dockerfile:experimental

FROM jetson-model-deployment

RUN pip3 install inputs
RUN pip3 install threading
RUN pip3 install pyrealsense2

# setup entrypoint
COPY entrypoint.sh /

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
