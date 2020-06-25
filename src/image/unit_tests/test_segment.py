from src.image.segment import get_border

import numpy as np

from numpy.testing import assert_array_equal


IMAGE = np.array([[0.24267094, 0.69918836, 0.73619987, 0.47642466],
       [0.84590931, 0.8760962 , 0.61977415, 0.32259561],
       [0.46912959, 0.48902284, 0.49063912, 0.6081179 ],
       [0.29475754, 0.89694127, 0.94615964, 0.14136022],
       [0.58346509, 0.65756213, 0.17460532, 0.2919921 ]])

BORDER = np.array([0.58346509, 0.29475754, 0.46912959, 0.84590931, 0.24267094,
       0.69918836, 0.73619987, 0.47642466, 0.32259561, 0.6081179 ,
       0.14136022, 0.2919921 , 0.17460532, 0.65756213])


def test_get_border():
    
    border = get_border(IMAGE)
    
    assert_array_equal(border, BORDER)