import unittest
from ImageProcessing import ImageProcessing


class TestImageProcessing(unittest.TestCase):
    """
     These tests might be a bit slow, becouse handling images.
    """
    
    def test_get_dictonary_without_noise(self):
        ip = ImageProcessing()
        image = ip.read_im('images/test_ocean.png')
        
        h = 32
        all_patches_n = ip.get_all_patches(h,image).shape[0]
        
        incomplete_patches_n = ip.get_incomplete_patches(image,h).shape[0]
        self.assertEqual(incomplete_patches_n,0)
        
        self.assertEqual(ip.get_dictionnary_patches(image, h).shape[0], all_patches_n)
        
    def test_get_dictonary_with_one_black_box(self):
        ip = ImageProcessing()
        image = ip.read_im('images/test_ocean.png')
        
        h = 32
        all_patches_n = ip.get_all_patches(h,image).shape[0]
        
        modified_image = ip.delete_rect(image, 25,25,10,10)
        
        incomplete_patches_n = ip.get_incomplete_patches(modified_image,h).shape[0]
        self.assertNotEqual(incomplete_patches_n,0)
        
        dictionnary_n = ip.get_dictionnary_patches(modified_image, h).shape[0]
        
        self.assertEqual(incomplete_patches_n + dictionnary_n, all_patches_n)

if __name__ == '__main__':
    unittest.main()