"""
Automated Integration and Verification Test for Fake News Detection App
"""

import json
import unittest
from app import app, best_model_key, metrics_data

class TestFakeNewsApp(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_homepage(self):
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b'VERITAS ML', resp.data)
        self.assertIn(b'Live Article Classifier', resp.data)
        self.assertIn(best_model_key.encode(), resp.data)

    def test_metrics_api(self):
        resp = self.client.get('/api/metrics')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('SVM', data['data']['models'])
        self.assertIn('Random Forest', data['data']['models'])
        self.assertIn('XGBoost', data['data']['models'])
        # Check actual calculated accuracy
        svm_acc = data['data']['models']['SVM']['accuracy']
        self.assertGreater(svm_acc, 0.90)

    def test_predict_fake_news(self):
        payload = {
            "title": "SHOCKING BOMBSHELL: Secret FBI Conspiracy To Rig General Election",
            "text": "Explosive undercover documents reveal an unprecedented conspiracy orchestrated by corrupt mainstream media moguls and political elites to deceive the public! Wake up, the entire election was completely rigged!!",
            "model": "SVM"
        }
        resp = self.client.post('/api/predict', json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['prediction']['label'], 'FAKE NEWS')
        self.assertGreater(data['prediction']['confidence_percent'], 50.0)
        
        # Check linguistic analysis
        ling = data['linguistic_analysis']
        self.assertGreater(ling['sensationalism_score'], 30)
        self.assertGreater(ling['exclamation_count'], 0)
        self.assertIn('conspiracy', [w.lower() for w in ling['sensational_words_found']])

    def test_predict_genuine_news(self):
        payload = {
            "title": "Kerry to visit Paris in gesture of sympathy after attacks",
            "text": "WASHINGTON (Reuters) - U.S. Secretary of State John Kerry will travel to Paris this week in a gesture of sympathy and solidarity with France following last week's deadly attacks, the State Department announced on Monday. Department spokesman John Kirby confirmed during a press briefing that Kerry will hold talks with French officials.",
            "model": "SVM"
        }
        resp = self.client.post('/api/predict', json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['prediction']['label'], 'GENUINE NEWS')
        self.assertGreater(data['prediction']['confidence_percent'], 50.0)

        # Check attribution score
        ling = data['linguistic_analysis']
        self.assertGreater(ling['attribution_score'], 30)
        self.assertIn('announced', [w.lower() for w in ling['attribution_words_found']])

    def test_all_models_inference(self):
        text = "Federal Reserve officials indicated Wednesday that interest rates are likely to remain unchanged."
        for m in ['SVM', 'Random Forest', 'XGBoost']:
            resp = self.client.post('/api/predict', json={"text": text, "model": m})
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertEqual(data['status'], 'success')
            self.assertIn(data['prediction']['label'], ['FAKE NEWS', 'GENUINE NEWS'])

if __name__ == '__main__':
    unittest.main()
