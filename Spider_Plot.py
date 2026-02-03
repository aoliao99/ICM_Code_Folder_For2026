import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

def radar_factory(num_vars, frame='circle'):
    """
    Angle and Axis Settings Required for Creating a Spider Plot
    """
    # Calculation angle: Divide the circumference (2π) into num_vars equal parts.
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'
        #  Rotate the figure so that the first axis is at the top.
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Rotate 90 degrees minus half of the first angle
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=0.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

# --- 1. Prepare Data (Data sourced from AHP-TOPSIS output) ---
# Indicator Selection：Social , Originality, Deductive
labels = ['Social\n(Human Connection)', 'Originality\n(Innovation)', 'Deductive\n(Critical Judgment)']

# Data entry 
# Format: [Social, Originality, Deductive]
data = {
    'Programmer (STEM)': [53, 50, 69], 
    'Chef (Trade)':      [50, 40, 59],
    'Singer (Arts)':     [53, 60, 47], 
    'Counselor (Benchmark)': [85, 60, 75] # 加入 Counselor 作为高社会价值对照组
}

# --- 2. Plot Settings ---
N = len(labels)
theta = radar_factory(N, frame='polygon')

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='radar'))
fig.subplots_adjust(top=0.85, bottom=0.05)

colors = {'Programmer (STEM)': '#1f77b4',  
          'Chef (Trade)': '#ff7f0e',        
          'Singer (Arts)': '#d62728',      
          'Counselor (Benchmark)': '#2ca02c'}

#AI Statement: This section utilizes Gemini3pro for code optimization.
# --- 3.  Draw each line---
# Set transparency and line width
for i, (title, case_data) in enumerate(data.items()):
    ax.plot(theta, case_data, color=colors[title], linewidth=2, label=title)
    ax.fill(theta, case_data, facecolor=colors[title], alpha=0.15) # 填充颜色

# --- 4. Enhance charts ---
ax.set_varlabels(labels)
ax.set_ylim(0, 100)
ax.grid(True, color='grey', alpha=0.3)
# Add Title
plt.title('Non-Economic Value Assessment\n(Identifying "Human-Centric" Worth beyond AI Risk)', 
          size=16, color='black', y=1.05, weight='bold')
# Add legend
legend = ax.legend(loc=(0.9, .95), labelspacing=0.1, fontsize='small')

# --- 5. Analytical Commentary  ---
plt.figtext(0.5, 0.02, 
            "Observation:\n"
            "1. Counselor & Singer show high distribution in 'Social' & 'Originality', indicating high Cultural/Ethical Value.\n"
            "2. Even if AI Risk is high, these 'Human-Centric' traits justify maintaining educational programs.", 
            ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.1, "pad":5})

plt.show()
