#!/bin/bash

# Script to perform all airflow related actions

set -xe

echo "starting script"

deploy_dags() {
    echo "starting with dag deployment"
    epctl login --client-id ${CLIENT_ID} --client-secret ${CLIENT_SECRET} --production
    epctl map sync-dags --region ${AWS_REGION} --production true --cluster-name $SCHEDULER --directory dags
}

rename_dags_and_variables() {
    echo "renaming dags and variables to add prefix"
    find dags -maxdepth 1 -not -type d -exec echo {} ';' | while read f; do
            base_name=$(basename $f) &&
            sed -i "s/dag_id \?= \?f\?\"/dag_id = f\""${BRANCH_NAME}"-/g" $f &&
            sed -i "s/Variable.get(\"${projectName2}_/Variable.get(\"${AIRFLOW_VARIABLE_PREFIX}${projectName2}_/g" $f && \
            mv "dags/$base_name" "dags/"${BRANCH_NAME}"-$base_name"
    done
}

cleanup_dags_and_variables() {
    echo "starting with dag cleanup"

    find dags -maxdepth 1 -not -type d -exec basename {} ';' | while read f; do
        echo "deleting dag: "$f
        epctl login --client-id ${CLIENT_ID} --client-secret ${CLIENT_SECRET} --production
        epctl map list-clusters --region ${AWS_REGION} --production true
        epctl map delete-dag --region ${AWS_REGION} --production true --cluster-name $SCHEDULER --file-name "$PR_NO-$f" || true
    done

    # Delete variables in airflow variables file
    envsubst <airflow-variables.csv | while IFS=, read -r key value || [ -n "$key" ]; do
        key=$(echo $key | tr -d '"')
        curl --location --request DELETE https://gateway.eu-west-1.map.nike.com/${SCHEDULER}/api/v1/variables/${PR_NO}-${key} \
            --header "Authorization: Bearer $access_token"
    done
}

cleanup_variables() {
    echo "starting with variable cleanup"
}

delete_schedule() {
    echo "removing schedule from airflow variables"
    sed -i "s/${projectName2}_schedule.*/${projectName2}_schedule,/g" ./airflow-variables.csv
}

deploy_airflow_variables() {

    # Remove schedule variable for non-prod env
    if [ $SCHEDULE_DAGS == 'false' ]; then
        delete_schedule
    fi
    # Replace variables in airflow variables file
    envsubst <airflow-variables.csv | while IFS=, read -r key value || [ -n "${key}" ]; do
        echo $key $value
        AIRFLOW_VAR=$(jq --null-input --arg airflow_key "$key" --arg airflow_value "$value" '{"key": $airflow_key, "value": $airflow_value}')
        # create variables
        curl --fail --location --request POST https://gateway.eu-west-1.map.nike.com/${SCHEDULER}/api/v1/variables \
            --header 'accept: application/json' \
            --header 'Content-Type: application/json' \
            --header "Authorization: Bearer $access_token" \
            --data "$AIRFLOW_VAR" \
            --retry 3
    done
}

build_and_deploy_to_ecr() {
    echo "building image and deloying to ECR"

    export DOCKER_CONFIG=$(pwd)
    $(aws ecr get-login --no-include-email --region eu-west-1)
    aws ecr describe-repositories --repository-names knnights/${projectName1} ||
        aws ecr create-repository --repository-name knnights/${projectName1} &&
        aws ecr set-repository-policy --repository-name knnights/${projectName1} --policy-text file://ecr-policy.json
    docker build -t knnights/${projectName1} .
    docker tag knnights/${projectName1} ${DOCKER_IMAGE}:${IMAGE_TAG}
    docker push ${DOCKER_IMAGE}:${IMAGE_TAG}
    docker tag knnights/${projectName1} ${DOCKER_IMAGE}:latest
    docker push ${DOCKER_IMAGE}:latest

}

deploy_sagemaker_pipeline() {
    poetry install
    export ENV=${ENVIRON}
    aws configure set default.region eu-west-1
    find . -type f -name 'sagemaker_pipeline.py' -o -name '*drift.py' -not -path ./shapeshifter/generic/sagemaker_pipeline.py | while read pipeline;
        do poetry run python ${pipeline}
    done
}

"$@"
