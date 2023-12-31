
pipeline {

    agent any


    stages {
        stage("CheckoutGit") {
            steps {
                    git credentialsId: 'git_creds', url: 'https://github.com/Taha-Cakir/jenkins-mlops-monitoring.git' , branch: 'main'
            }
        }
        stage("InstallLibs") {
            steps {
                sh """
                pip install boto3 sagemaker evidently pandas numpy s3fs
                
                
                """
            }
        }

        // --commit-id $COMMIT_HASH    
        stage("Run Monitoring Job") {
            steps {
              sh """
              
                python3 evidently.py --baseline $baseline --test $test 
                

              """
            }
        }
    }
}

/*        
        stage("TrainModel") {
            steps { 
              sh """
               aws sagemaker create-training-job --training-job-name ${params.SAGEMAKER_TRAINING_JOB}-${env.BUILD_ID} --algorithm-specification TrainingImage="${params.ECRURI}:${env.BUILD_ID}",TrainingInputMode="File" --role-arn ${params.SAGEMAKER_EXECUTION_ROLE_TEST} --input-data-config '{"ChannelName": "training", "DataSource": { "S3DataSource": { "S3DataType": "S3Prefix", "S3Uri": "${params.S3_TRAIN_DATA}"}}}' --resource-config InstanceType='ml.c4.2xlarge',InstanceCount=1,VolumeSizeInGB=5 --output-data-config S3OutputPath='${S3_MODEL_ARTIFACTS}' --stopping-condition MaxRuntimeInSeconds=3600
              """
             }
        }

      stage("TrainStatus") {
            steps {
              script {
                    def response = sh """ 
                    aws lambda invoke --function-name ${params.LAMBDA_CHECK_STATUS_TRAINING} --cli-binary-format raw-in-base64-out --region us-east-1 --payload '{"TrainingJobName": "${params.SAGEMAKER_TRAINING_JOB}-${env.BUILD_ID}"}' response.json
                    sleep 240
                    """
                    
                  }
              }
      }

      stage("DeployToTest") {
            steps { 
              sh """
               aws sagemaker create-model --model-name ${params.SAGEMAKER_TRAINING_JOB}-${env.BUILD_ID}-Test --primary-container ContainerHostname=${env.BUILD_ID},Image=${params.ECRURI}:${env.BUILD_ID},ModelDataUrl='${S3_MODEL_ARTIFACTS}'/${params.SAGEMAKER_TRAINING_JOB}-${env.BUILD_ID}/output/model.tar.gz,Mode='SingleModel' --execution-role-arn ${params.SAGEMAKER_EXECUTION_ROLE_TEST}
               aws sagemaker create-endpoint-config --endpoint-config-name ${params.SAGEMAKER_TRAINING_JOB}-${env.BUILD_ID}-Test --production-variants VariantName='single-model',ModelName=${params.SAGEMAKER_TRAINING_JOB}-${env.BUILD_ID}-Test,InstanceType='ml.m4.xlarge',InitialVariantWeight=1,InitialInstanceCount=1
               aws sagemaker create-endpoint --endpoint-name ${params.SAGEMAKER_TRAINING_JOB}-${env.BUILD_ID}-Test --endpoint-config-name ${params.SAGEMAKER_TRAINING_JOB}-${env.BUILD_ID}-Test
               sleep 300
              """
             }
        }

      stage("SmokeTest") {
            steps { 
              script {
                 def response = sh """ 
                 aws lambda invoke --function-name ${params.LAMBDA_EVALUATE_MODEL} --cli-binary-format raw-in-base64-out --region us-east-1 --payload '{"EndpointName": "'${params.SAGEMAKER_TRAINING_JOB}-${env.BUILD_ID}'-Test", "Body": {"Payload": {"S3TestData": "${params.S3_TEST_DATA}", "S3Key": "test/iris.csv"}}}' evalresponse.json
              """
              }
            }
        }

      stage("DeployToProd") {
            steps { 
              sh """
               aws sagemaker create-model --model-name ${params.SAGEMAKER_TRAINING_JOB}-${env.BUILD_ID}-Prod --primary-container ContainerHostname=${env.BUILD_ID},Image=${params.ECRURI}:${env.BUILD_ID},ModelDataUrl='${S3_MODEL_ARTIFACTS}'/${params.SAGEMAKER_TRAINING_JOB}-${env.BUILD_ID}/output/model.tar.gz,Mode='SingleModel' --execution-role-arn ${params.SAGEMAKER_EXECUTION_ROLE_TEST}
               aws sagemaker create-endpoint-config --endpoint-config-name ${params.SAGEMAKER_TRAINING_JOB}-${env.BUILD_ID}-Prod --production-variants VariantName='single-model',ModelName=${params.SAGEMAKER_TRAINING_JOB}-${env.BUILD_ID}-Prod,InstanceType='ml.m4.xlarge',InitialVariantWeight=1,InitialInstanceCount=1
               aws sagemaker create-endpoint --endpoint-name ${params.SAGEMAKER_TRAINING_JOB}-${env.BUILD_ID}-Prod --endpoint-config-name ${params.SAGEMAKER_TRAINING_JOB}-${env.BUILD_ID}-Prod
              """
             }
        }

  }
  */
