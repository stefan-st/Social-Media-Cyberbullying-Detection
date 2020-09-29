# Automatic detection of cyberbullying on social media platforms

An automatic detection model for cyberbullying posts that is not biased towards a specific social media platform or a certain type of bullying. 
The classifier for predicting the probability of a post to be cyberbullying is composed of two different classifiers trained on datasets collected from Twitter and Formspring.

The features used for training the classifieres be organized in 4 different categories, that are further explained in the [paper](paper/CyberbullyingDetectionPaperRoCHI.pdf):
- content features
- subjectivity features
- aggressivity features
- general content features.

For testing, we collected post samples from twitter, that can be seen in the `twitter_samples` folder. 
Finally, after the trained classifier was used for predicting the cyberbullying probability of the posts in the samples, we used human classifiers for validating the results.
The results can be inspected in the `results` folder where a spreadsheet with the post text along with the score given to it by the classifier and the manual annotators can be seen.

**To cite this work, please use:**
<br/>
Stan, S. and Rebedea, T. (2020). Automatic detection of cyberbullying on social media platforms. In Proceedings of RoCHI - International Conference on Human-Computer Interaction 2020, Matrix ROM.
