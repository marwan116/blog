---
title: AWS IAM management concepts and gotchas
date: 2023/04/20
description: A detailed guide to managing access to AWS resources using IAM.
tag: cloud, aws
author: Marwan
---

# Brief

This article contains a summary of learnings about AWS Identity and Access Management (IAM) management from the perspective of a small engineering team at an early-stage startup.

Being part of a small team almost certainly means having to wear many hats, one of which is defining and managing management for other team members, pipelines (CI/CD and data pipelines), services, data stores, and other resources.


# IAM

Let's explain some IAM concepts right off the bat. IAM consists of the following:

- entities:
    - users: individual entities that can be assigned AWS credentials (access key ID and secret access key) allowing them to access AWS programmatically or via the AWS console (i.e. the browser).
    - groups: a collection of users that can be assigned policies which grants the group's users access to AWS resources.
    - roles: they are similar to users, but they are not associated with a specific person. Roles are created to define a set of permissions that can be assumed by an entity like an AWS service or user.

- policies:
    - Resource-based policies: 
        - managed policies: standalone policies that you can attach to multiple users, groups, and roles in your AWS account. Their is a default set of managed policies provided by AWS which you can extend.
        - inline policies: inline policies are policies that you create within a specific entity (user or group). They are embedded directly into the entity to which they apply and can't be shared with other entities.
    - Trust policies: a trust policy is grants an entity permission to assume a role. For example, you can attach a trust policy to a role that allows an AWS service to assume the role.

## Entities

### Users
A User is provided with at least one of the following credentials:
- a password to allow AWS console access
- access keys to allow programmatic access

By default, a brand new IAM user created using the AWS CLI or AWS API has no credentials of any kind. You must create the type of credentials for an IAM user based on your use case. Additionally as a best practice it is best to enable multi-factor authentication (MFA) for the IAM user. 

Note that access keys are permanent (i.e. can't set an expiry date on). More on the security ramifications of permanent credentials below.

### Groups
A group can be created from the IAM console and can be assigned a set of policies. Users can then be added to the group and they will inherit the policies assigned to the group.

Some things to note about groups:
* A user group can contain many users, and a user can belong to multiple user groups.
* User groups can't be nested; they can contain only users, not other user groups.
* There is no default user group that automatically includes all users in the AWS account. If you want to have a user group like that, you must create it and assign each new user to it.
* The number and size of IAM resources in an AWS account, such as the number of groups, and the number of groups that a user can be a member of, are limited. For more information, see IAM and AWS STS quotas.

If an IAM user belongs to multiple user groups, then the user's permissions can in most cases be simplified as the union of the permissions granted by all the user groups to which the user belongs. For example, if one group grants the user permission to perform an Amazon EC2 action and another group grants the user permission to perform an Amazon S3 action, the user belongs to both groups, and the user can perform both EC2 and S3 actions.

The exception to this rule is when one group grants an action and another group explicitly denies the same action to the user. An explicit deny overrides any allow. In this case, the user is denied the permission to perform the action.


### Roles
A role provides temporary access to AWS resources. Roles are not associated with a specific user or group. Instead, trusted entities assume roles, such as IAM users, applications, or AWS services such as EC2.

Roles were originally introduced following a tranditional Role-Based Access Control (RBAC) model. However, roles have evolved to be used in a number of different ways (lookup attribute-based access control (ABAC) using IAM roles and tags).

In the context of this article, we will focus on simple use cases of roles, which is to grant temporary access to AWS resources.


## Policies:

### Resource-based policies
A resource-based policy is a document that looks like this:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "...",
            "Effect": "...",
            "Action": "...",
            "Resource": "...",
            "Condition": "..."
        }
    ]
}
```

Where:
- Sid: a unique identifier for the policy statement. Usually a human-readable string is used like "S3AllowMyBucketRead"
- Effect: "Allow" or "Deny" access to the resource.
- Action: the action that you want to allow or deny. For example, "s3:GetObject" enables you to read objects from an S3 bucket.
- Resource: Is a pattern matching one or more AWS resource arns. For example, "arn:aws:s3:::my-bucket/*" enables you to read objects from the my-bucket S3 bucket.
- Condtion: conditions are optional but they can be used for finer grained control over access to resources. For example, you can restrict access to a resource based on the time of day, IP address, or whether the request was made over SSL.

### Trust policies
A trust policy is a document that grants an entity permission to assume a role. For example, you can attach a trust policy to a role that allows an AWS service to assume the role.

A trust policy is a document that looks like this:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "...",
            "Effect": "...",
            "Principal": {
                "AWS": "...",
                "Service": "...",
                "Federated": "...",
                "Condition": "..."
            },
            "Action": "sts:AssumeRole",
            "Condition": "..."
        }
    ]
}
```

Where:

`Principal`: defines the entity that can assume the role. The principal can be an AWS account (AWS), a federated user (Federated), or an AWS service (Service). For example, you can specify the Amazon EC2 service principal (ec2.amazonaws.com) to allow Amazon EC2 instances to assume a role. A `Condition` can also be specified to further restrict the entity that can assume the role. For example, you can specify the IP address range of the entity that can assume the role.

The "Action": "sts:AssumeRole" is what defines the trust policy. This is the only action that you can specify in a trust policy.


# IAM setup journey

### The IAM Hotel setup analogy
This is adapted from an [IAM announcement in August 2011](https://aws.amazon.com/blogs/aws/aws-identity-and-access-management-now-with-identity-federation/)

Imagine that your AWS account is like a hotel and you are the innkeeper. At the start of your AWS journey, you are given root access - i.e. you are offered a single master key granting unconditional access to all of your AWS resources.

With more than one user, you can proceed to share the master key (root account) with your employees. This is not ideal, for more than one reason (everyone needs to be notified when the key is updated, if one of your employees leaves you will need to update the key, and if compromised the entire AWS account is at risk). Or you can create the equivalent of access badges for your hotel employees by creating IAM users and groups - i.e. housekeeping gets access to guest rooms, cooks get access to the kitchen, in effect allowing you to grow your business while giving you explicit control over the permissions granted to your employees.

Lets think of your hotel guests. You could give them the same access badges as you give to your employees, but since their stay is short you would need to ensure that they return their badges when they leave.To solve this, you enable your front desk to issue temporary hotel access cards to your guests for accessing only their room and hotel facilities. IAM roles are the equivalent of those temporary hotel access cards.

## Starting with a root account
When you first create an AWS account, you are assigned as the root user of the account. The root user has complete access to all resources in the account. As a security best practice you should not use the root user for everyday tasks, even the AWS console will warn you about this. The root user should only be used to create your first IAM user. Make sure you set multi-factor authentication (MFA) on the root user.

## Expanding to more users

A typical IAM setup of a new user would be:

1. Create a new user.
2. Assign the user to a group.
3. Assign the group a set of resource-based policies.
4. Create an access key for the user.
5. Provide the user with the access key ID and secret access key.

Adopting this approach means that you can easily manage permissions for multiple users by simply adding or removing them from groups. This is a much better approach than assigning policies directly to users.

### Concept of least privilege

To ensure that your AWS resources are secure, it is important to follow the principle of least privilege. This means that you should only grant users the permissions that they need to perform their job and no more. This is important because if a user's credentials are compromised, the attacker will only have access to the resources that the user has access to.

#### Approach 1: Start with the closest default AWS managed policy and taylor it
Let's say you want to figure out a policy for read-only access to a given S3 prefix.

You can:
- Clone the AWS provided managed policy AmazonS3ReadOnlyAccess
- Start editing the policy by restricting it to your prefix
- Confirm the policy is still working as expected by running tests after each modification

Figuring out read-only access to an S3 prefix is fairly trivial, this applies to other services as well.

#### Approach 2: Use a tool that relies on client-side monitoring to detect your required permissions
There is an open source tool named [iamlive](https://github.com/iann0036/iamlive) which allows you to run a command and then generates a policy based on the AWS API calls made by the command. This is not a perfect tool, but it can be useful to get started.


#### Approach 3: Rely on cloudtrail and AWS access analyzer to infer permissions after usage
This is an approach recommended by AWS. It relies on the following tools:
- [Cloudtrail](https://aws.amazon.com/cloudtrail/) to log all API calls made to AWS
- [AWS access analyzer](https://aws.amazon.com/access-analyzer/) to analyze the Cloudtrail logs and generate a policy based on the API calls made

This approach is more mature than the previous ones, but it requires more setup:
- It suggests providing a given user or service with very permissive permissions
- Then relying on Cloudtrail to audit the API calls made
- Then relying on AWS access analyzer to generate a policy based on the API calls made

### Enforcing and automating rotation

One major drawback of using access keys is that they cannot be rotated automatically. They are permanent keys by default. It is a security best practice to rotate access keys regularly. This is because if a key is compromised, it can be used to access your AWS resources until the key is rotated.

To deny access to users who have not rotated their access keys within a certain period of time, you can attach the following policy to the user/group:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "RotateAccessKeys",
            "Effect": "Deny",
            "Action": "*",
            "Resource": "*",
            "Condition": {
                "DateGreaterThan": {
                    "aws:AccessKeyLastRotated": "90 days ago"
                }
            }
        }
    ]
}
```

We have found the following open source tool useful for automating the rotation of access keys:
https://github.com/rhyeal/aws-rotate-iam-keys#configuration

## Creating roles

### Use Case: Kubernetes service accounts

When working in kubernetes, one might need to allow a deployment access to AWS resources like (s3, glue, athena, ...). 

One naiive approach is to build the AWS credentials as secrets in the deployment. This is not ideal from a security perspective, as the credentials are now stored in the cluster and can be compromised and are difficult to rotate.

Another approach is to make use an integration between AWS Secrets Manager and Kubernetes (EKS). This is a better approach, but it still requires the credentials to be stored in the cluster and it requires making use of a new service (AWS Secrets Manager).

The preferred approach that we found is to associate kubernetes service accounts with IAM roles (See this guide in [the documentation here](https://docs.aws.amazon.com/eks/latest/userguide/iam-roles-for-service-accounts.html))

#### Using a gitlab runner on EKS to run a job that requires access to AWS resources
This is a common use case. You might want to run a CI pipeline on gitlab that requires access to AWS resources. You can use the same approach as above to associate a kubernetes service account with an IAM role. Then you specify the CI job to use the gitlab runner that is deployed on EKS. This way the job will have access to the AWS resources that the IAM role has access to.

See [this guide here](https://dev.to/stack-labs/securing-access-to-aws-iam-roles-from-gitlab-ci-3an0)

