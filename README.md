# Project-3
# Project-3: 💎Diamonds Showcase💎

## Overview 🌟

Welcome to the Diamonds Showcase! This project is part of the AI Bootcamp at OSU, where we delve into creating an interactive chatbot using cutting-edge technologies. In this project, we're leveraging Gradio for the user interface, integrating it with OpenAI for natural language processing, and DALL-E for image generation. Our goal is to build a conversational AI system that can provide information and generate images related to diamonds. 💬Did your boo thang get you a diamond that you want to learn more about? Come check out our chatbot!!!!!🔍🖼️

## Methodology

- Programming language: Python with source code editor, Visual Studio Code
- Data Sceince and Machine Learning: NumPy, Pandas, scikit-learn, PyTorch, TensorFlow, KerasTuner
- Additional libraries: seaborn, statsmodels, matplotlib.pyplot, plotly.express, and torch

## Features 🚀

- **Chat Interface**: Utilize Gradio to create an intuitive chat interface for users to interact with the chatbot.
- **Natural Language Understanding**: Integrate OpenAI's language model to understand and respond to user queries naturally.
- **Image Generation**: Leverage DALL-E's capabilities to generate images based on user requests related to diamonds.
- **Information Retrieval**: Provide users with information about diamonds, including pricing, characteristics, and more. 📊💡

## Dependencies 📦

- Gradio
- OpenAI API
- DALL-E API
- Python 3.x

## Installation 💻

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/project-3-diamonds-chatbot.git
   ```

2. Install dependencies:

   ```bash
   pip install gradio openai dalle-py
   ```

3. Run the application:

   ```bash
   python app.py
   ```

4. Access the chatbot interface through your web browser at `http://localhost:7860`.

## Usage 🎮

1. Enter your query or request in the chat window.
2. The chatbot will process your input, provide relevant information, and generate images as needed.
3. Enjoy conversing with the Diamonds Chatbot! 💬🎉

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
- The regression models need additional testing with adjustments to hyperparameters since it still shows overfitting despite optimizations 
- With PyTorch, more research on how to enhance the model, such as adjusting the model architecture by adding more layers, incresing the number of neurons or different activation functions. Implement and experiment with different regularization techniques to prevent overfitting and improve generalization performance. This could include dropout regularization, L2 regularization (weight decay), or early stopping.




## Conclusion

Summarize the key findings and conclusions of the chatbot project, highlighting its significance, impact, and value proposition. Emphasize any achievements, successes, or lessons learned during the development process.



## Contributing 🤝

Contributions are welcome! Feel free to submit issues or pull requests if you have any ideas for improvements or encounter any bugs.

## License 📜

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements 🙏

We would like to thank the developers of Gradio, OpenAI, and DALL-E for providing the tools and APIs that make this project possible. 👏🌟
