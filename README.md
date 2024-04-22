# Project-3
💎**Diamonds Showcase README**💎

[**Presentation**](https://docs.google.com/presentation/d/180iazh-0e2mQfu0h9xBafVOfzKl_t-bgL2INx06a48A/edit#slide=id.g26f1b39ef97_1_23)

## Overview 🌟

Welcome to the Diamonds Showcase! This project is part of the AI Bootcamp at OSU, where we delve into creating an interactive diamond-centric experience using cutting-edge technologies. In this project, we're leveraging Gradio for the user interfaces, integrating CHATGPT for natural language processing, and DALL-E for image generation. Our goal is to build an interactive, comprehensive AI-based experience for those interested in diamonds. 💬Did your boo thang get you a diamond that you want to learn more about? Come check out our project!!!!!🔍🖼️

## Methodology

- Programming language: Python with source code editor, Visual Studio Code
- Data Science and Machine Learning: NumPy, Pandas, scikit-learn, PyTorch, TensorFlow, KerasTuner
- Additional libraries: seaborn, statsmodels, matplotlib.pyplot, plotly.express, and torch

## Features 🚀

- **Gradio Interfaces**: Utilize Gradio to create an intuitive interface for users to interact with AI-based tools
- **Natural Language Understanding**: Integrate OpenAI's language model to understand and respond to user queries naturally
- **Deep Neural Network Based Price Prediction**: Utilize both TensorFlow and PyTorch to create highly accurate 
- **Image Generation**: Leveraged DALL-E's capabilities to generate images based on user requests related to diamonds.
- **Information Retrieval**: Provide users with information about diamonds, including pricing, characteristics, and more. 📊💡

## Dependencies 📦

- Gradio
- OpenAI API
- DALL-E API
- Python 3.x

## Usage 💻🎮

1. Clone this repository
2. Install dependencies
3. Run all .ipynb notebooks
4. Update local gradio URLs in website.html file
5. Run website
6. Enter your query or request in desired interface
7. Enjoy 💬🎉

## Pinecone 
Utilizing Pinecone and vectorizing text into chunks are crucial steps in improving the performance and efficiency of NLP text summarizers. Pinecone is a vector database service designed for handling high-dimensional vector data at scale, offering efficient storage and retrieval of vectorized data. Vectorizing text into chunks reduces dimensionality and preserves semantic information, allowing Pinecone to efficiently search for and retrieve the most relevant information. These techniques combined contribute to building a fast, efficient, and accurate NLP text summarization system.

## Text to Image Generator 
Integrating a text-to-image generator with Gradio and DALL-E in Python is crucial for creating a powerful and interactive AI system. By combining these technologies, we can generate images directly from textual input, significantly enhancing user experience and versatility. Gradio provides an intuitive user interface, allowing users to input text and receive generated images seamlessly. DALL-E, on the other hand, is a state-of-the-art model capable of creating high-quality images from textual descriptions. This integration not only provides a visually enriched experience but also allows for a more comprehensive interaction, making it an indispensable tool for various applications, from creative design to content generation, and more.
[View image](/diamonds/image1.png)

## Training Data and Model
- For price prediction, an initial pipeline created for models’ Linear Regression, KNeighbors, RandomForest, ExtraTrees regressor and ADAboost, where ExtraTrees performed the best with R2 train/test scores of  0.9818/0.9999 and MAE train/test 0.4004/263.3307. Next, explored neural network performance by optimizing hyperparameters using Regression Hyper Model and Keras Tuner, to see if corrections can be made to the performance that is showing overfitting. 
- PyTorch framework was also used to predict price which included tensor creation, model definition, dataset creation, dataloading and model training. The first epoch indicates that the model's prediction is off by 24.36 from the actual target variable. Subsequesnt epochs show flucuations in the loss value which can indicate performance temporarialy deterioating. However, the loss gradually decreases, indicating improvement in the models's performance, ending with the last epoch at 0.00839. 
- The diamonds dataset went through preprocessing before the machine learning models were utilized.  This included dropping and unnamed column, checking and dropping null and NaaN values, checking data types for objects and converting the obects to numerical values using Label Encoder. 

## Implementation Details


Provide a high-level overview of the code structure, architecture, and design patterns used in the chatbot implementation. Highlight any innovative or noteworthy aspects of the codebase.



## Deployment and Integration

Describe how the chatbot is deployed and integrated into the target environment or platform. This may include deployment to web servers, cloud platforms (e.g., AWS, Azure), messaging platforms (e.g., Slack, Facebook Messenger), or mobile applications



# Future Work and Recommendations
- Host webpage on internet
- Expand text data available to chatbot
- Assess the highly performing deep neural network created with TensorFlow on outside data
- With PyTorch, more research on how to enhance the model, such as adjusting the model architecture by adding more layers, incresing the number of neurons or different activation functions. Implement and experiment with different regularization techniques to prevent overfitting and improve generalization performance. This could include dropout regularization, L2 regularization (weight decay), or early stopping.

## Contributions 🤝

Contributions are welcome! Feel free to submit issues or pull requests if you have any ideas for improvements or encounter any bugs.

## License 📜

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements and Citations 🙏
Diamond Analysis: Diamonds In-Depth Analysis (kaggle.com)
Ogden, J. (2018). Diamonds : An Early History of the King of Gems. Yale University Press. https://doi.org/10.12987/9780300235517
Aharonovich, I., Greentree, A. D., & Prawer, S. (2011). Diamond photonics. Nature Photonics, 5(7), 397–405. https://doi.org/10.1038/nphoton.2011.54
Types of Diamond Cuts - How to Choose The Right Shape – Padis Jewelry (padisgems.com)
Diamond Color Scale Chart | 4C's Education (rarecarat.com)
Diamond Color and Clarity – Andrea Bonelli

We would like to thank the developers of Gradio, OpenAI, and DALL-E for providing the tools and APIs that make this project possible. 👏🌟
