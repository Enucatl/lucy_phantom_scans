import click
import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

import dpc_reconstruction.phase_stepping as phase_stepping

base = "/sls/X02DA/data/e13510/Data20/FPD/Matteo/"
folders = [
    "170822.112312108833",
    "170822.112348398976",
    "170822.112424171975",
    "170822.112500206143",
    "170822.112536242490",
    "170822.112612307955",
    "170822.112648294657",
    "170822.112724318830",
    "170822.112800365217",
    "170822.112836380126",
    "170822.112912496169",
    "170822.112948408946",
    "170822.113025463414",
    "170822.113101577082",
    "170822.113137565768",
    "170822.113213594116",
    "170822.113249585752",
    "170822.113325616666",
    "170822.113401696272",
    "170822.113437684507",
    "170822.113513772456",
    "170822.113549735974",
    "170822.113625763774",
    "170822.113701937967",
    "170822.113737828380",
    "170822.113813911950",
    "170822.113849884212",
    "170822.113926943832",
    "170822.114003200221",
    "170822.114038960484",
    "170822.114114999013",
    "170822.114151166638",
    "170822.114227167255",
    "170822.114303081126",
    "170822.114339156619",
    "170822.114415162319",
    "170822.114451174059",
    "170822.114527209137",
    "170822.114603221263",
    "170822.114639311695",
    "170822.114715330778",
    "170822.114751334939",
    "170822.114827352054",
    "170822.114903385488",
    "170822.114939458779",
    "170822.115016457284",
    "170822.115052470440",
    "170822.115128513261",
    "170822.115204537285",
    "170822.115240608825",
    "170822.115316609243",
    "170822.115352622949",
    "170822.115428642655",
    "170822.115504715113",
    "170822.115540707049",
    "170822.115616746410",
    "170822.115652800696",
    "170822.115728804220",
    "170822.115804826907",
    "170822.115840861883",
]

@click.command()
def main():
    previous_array = 0
    absorption_dir = os.path.join(base, "stability/absorption")
    darkfield_dir = os.path.join(base, "stability/darkfield")
    os.makedirs(absorption_dir, exist_ok=True)
    os.makedirs(darkfield_dir, exist_ok=True)
    for i, folder in enumerate(tqdm(folders)):
        filenames = sorted(
            glob.glob(
                os.path.join(base, folder, "*.tif")
            )
        )
        images = [Image.open(x) for x in filenames]
        arrays = [np.array(x) for x in images]
        current_array = np.dstack(arrays)[100:, :, :]
        if not isinstance(previous_array, int):
            with tf.Session() as sess:
                samples = current_array
                flats = previous_array
                sample_tensor = tf.placeholder(
                    tf.float32, shape=samples.shape)
                flat_tensor = tf.placeholder(
                    tf.float32, shape=flats.shape)
                sample_signals = phase_stepping.get_signals(sample_tensor)
                flat_signals = phase_stepping.get_signals(flat_tensor)
                dpc_reconstruction = phase_stepping.compare_sample_to_flat(
                    sample_signals,
                    flat_signals
                )
                visibility = phase_stepping.visibility(flat_signals)
                dpc_np, flat_np, visibility_np = sess.run(
                    [
                        dpc_reconstruction,
                        flat_signals,
                        visibility,
                    ],
                    feed_dict={
                        sample_tensor: current_array,
                        flat_tensor: previous_array,
                    })
                absorption = Image.fromarray(dpc_np[..., 0])
                darkfield = Image.fromarray(dpc_np[..., 2])
                absorption_file_name = os.path.join(
                    absorption_dir, "{0}.tif".format(i))
                darkfield_file_name = os.path.join(
                    darkfield_dir, "{0}.tif".format(i))
                absorption.save(absorption_file_name)
                darkfield.save(darkfield_file_name)
        previous_array = current_array

if __name__ == "__main__":
    main()
