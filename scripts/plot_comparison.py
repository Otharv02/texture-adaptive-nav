import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from rclpy.serialization import deserialize_message
from px4_msgs.msg import VehicleLocalPosition

def read_bag(bag_path):
    db_path = f"{bag_path}/{bag_path.split('/')[-1]}_0.db3"
    conn    = sqlite3.connect(db_path)
    cursor  = conn.cursor()
    cursor.execute("SELECT data FROM messages")
    rows = cursor.fetchall()
    conn.close()

    x_list, y_list, z_list, t_list = [], [], [], []

    for row in rows:
        msg = deserialize_message(row[0], VehicleLocalPosition)
        x_list.append(msg.x)
        y_list.append(msg.y)
        z_list.append(-msg.z)  # NED → positive up
        t_list.append(msg.timestamp / 1e6)  # microseconds → seconds

    # Normalize time to start at 0
    t0 = t_list[0]
    t_list = [t - t0 for t in t_list]

    return np.array(t_list), np.array(x_list), np.array(y_list), np.array(z_list)


# Read both bags
t_base, x_base, y_base, z_base = read_bag("baseline_default")
t_adap, x_adap, y_adap, z_adap = read_bag("adaptive_default")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle("Baseline vs Adaptive Optical Flow — Position Comparison", fontsize=14)

axes[0].plot(t_base, x_base, label="Baseline", color="red",  linewidth=2)
axes[0].plot(t_adap, x_adap, label="Adaptive", color="blue", linewidth=2)
axes[0].set_ylabel("X Position (m)")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(t_base, y_base, label="Baseline", color="red",  linewidth=2)
axes[1].plot(t_adap, y_adap, label="Adaptive", color="blue", linewidth=2)
axes[1].set_ylabel("Y Position (m)")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(t_base, z_base, label="Baseline", color="red",  linewidth=2)
axes[2].plot(t_adap, z_adap, label="Adaptive", color="blue", linewidth=2)
axes[2].set_ylabel("Altitude (m)")
axes[2].set_xlabel("Time (seconds)")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig("comparison_plot.png", dpi=150)
plt.show()
print("Plot saved as comparison_plot.png")

