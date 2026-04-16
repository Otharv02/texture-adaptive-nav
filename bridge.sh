#!/bin/bash 
WORLD_NAME=$(gz topic -l | grep "^/world/" | head -n 1 | cut -d'/' -f3)

if [ -z "$WORLD_NAME" ]; then
    echo "No world detected!"
    exit 1
fi

echo "Detected world: $WORLD_NAME"

# Run bridge
ros2 run ros_gz_bridge parameter_bridge \
/world/$WORLD_NAME/model/x500_flow_0/link/flow_link/sensor/flow_camera/image@sensor_msgs/msg/Image@gz.msgs.Image