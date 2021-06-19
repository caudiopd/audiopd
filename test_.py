import unittest
import warnings 
warnings.filterwarnings('ignore')
from app import app

class BasicTestCase(unittest.TestCase):
    
    def test_login(self):
        tester = app.test_client(self)
        response = tester.get('/', content_type='html/text')
        self.assertEqual(response.status_code, 200)
    
    def test_profile(self):
        tester = app.test_client(self)
        response = tester.get('profile.html', content_type='html/text')
        self.assertEqual(response.status_code, 200)
    
    def test_index(self):
        tester = app.test_client(self)
        response = tester.get('index.html', content_type='html/text')
        self.assertEqual(response.status_code, 200)

    def test_predict(self):
        tester = app.test_client(self)
        response = tester.get('y_predict', content_type='html/text')
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
