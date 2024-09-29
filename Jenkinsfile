pipeline {
    agent any

    environment {
        PYTHONPATH = "/Users/saahil/Desktop/Coding_Projects/DL/smart_crop_disease_detection/src:$PYTHONPATH"
        DOCKER_IMAGE = 'smart_crop_disease_detection'
        PATH = "/usr/local/bin:$PATH"  // Add Docker's path to the environment
    }

    stages {
        stage('Checkout') {
            steps {
                // Pull the latest code from GitHub
                git branch: 'main', url: 'https://github.com/saahil1801/Smart-Crop-Disease-Detection-using-PL.git'
            }
        }

        stage('Create Virtual Environment') {
            steps {
                // Install Python3 via Homebrew if necessary
                sh 'which python3 || brew install python'

                // Create a Python virtual environment
                sh 'python3 -m venv venv'
                
                // Upgrade pip inside the virtual environment
                sh '. venv/bin/activate && pip install --upgrade pip'
            }
        }

        stage('Install Dependencies') {
            steps {
                // Activate the virtual environment and install dependencies from requirements.txt
                sh '. venv/bin/activate && pip install -r requirements.txt'
            }
        }

        stage('Install DVC and Pull Data') {
            steps {
                // Activate the virtual environment, install DVC, and pull data
                sh '. venv/bin/activate && pip install dvc'
                sh '. venv/bin/activate && dvc pull'
            }
        }

        stage('Run Tests') {
            steps {
                // Set the correct PYTHONPATH and run tests using pytest
                sh '. venv/bin/activate && PYTHONPATH=$PYTHONPATH pytest'
            }
        }

        stage('Build Docker Compose') {
            steps {
                // Build the Docker containers with Docker Compose
                sh 'docker-compose -f docker-compose.yml build'
            }
        }

        stage('Stop Existing Docker Containers') {
            steps {
                // Stop any running container on port 5001, if exists
                script {
                    def containerId = sh(script: "docker ps -q --filter 'publish=5001'", returnStdout: true).trim()
                    if (containerId) {
                        sh "docker stop $containerId"
                        echo "Stopped existing container with ID: $containerId"
                    } else {
                        echo "No existing container running on port 5001"
                    }
                }
            }
        }

        stage('Deploy Application with Docker Compose') {
            steps {
                // Deploy the containers with Docker Compose
                sh 'docker-compose -f docker-compose.yml up -d'
            }
        }
    }

    post {
        always {
            // Clean up the workspace after the pipeline is done
            cleanWs()
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
}
