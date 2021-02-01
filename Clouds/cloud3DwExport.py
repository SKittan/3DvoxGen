from CAclouds import CAclouds3D
import numpy as np
import vispy.scene
from vispy.scene import visuals
from vispy.util import keys

import os

from pyvox.models import Vox
from pyvox.writer import VoxWriter

cloud = CAclouds3D(100, 100, 100, 'cuda')

cloud.init_elliptic_probabilities(50, 50, 50, 5., 5., 1., 100, 1, 500, 10., 10.)
cloud.simulate(20)

xyz = cloud.get_cloud_positions()

# By: https://github.com/michaelaye/vispy/blob/master/examples/basics/scene/point_cloud.py
# Make a canvas and add simple view
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
# create scatter object and fill in the data
scatter = visuals.Markers()
scatter.set_data(xyz.cpu().numpy(), edge_color=None, face_color=(1, 1, 1, .5), size=5)

view.add(scatter)
view.camera = 'turntable'  # or try 'arcball'

@canvas.events.key_press.connect
def on_key_press(event):
    if event.key is keys.SPACE:
        cloud.step()
        xyz = cloud.get_cloud_positions()
        scatter.set_data(xyz.cpu().numpy(), edge_color=None, face_color=(1, 1, 1, .5), size=5)
        canvas.update()

if __name__ == '__main__':
    import sys
    if sys.flags.interactive != 1:
        vispy.app.run()
        xyz = cloud.cld.cpu().numpy()
        coordinates_xyz = np.argwhere(xyz)
        x_min, y_min, z_min = coordinates_xyz.min(axis=0)
        x_max, y_max, z_max = coordinates_xyz.max(axis=0)
        xyz = xyz[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]

        vox = Vox.from_dense(xyz)

        numVoxFiles = 0
        for content in os.listdir():
            if content.endswith('vox'):
                numVoxFiles += 1

        VoxWriter('Cloud_' + str(numVoxFiles) + '.vox', vox).write()