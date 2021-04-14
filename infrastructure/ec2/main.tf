provider "aws" {
  region = "us-east-2"
}

locals {
  user_data = templatefile("${path.module}/userdata.yaml", {
    user1     = var.user1,
    user2     = var.user2,
    user3     = var.user3,
    user1_key = var.user1_key,
    user2_key = var.user2_key,
    user3_key = var.user3_key
  })
  tags = {
    Project = "CS598"
  }
}

data "terraform_remote_state" "network" {
  backend = "s3"

  config = {
    bucket = "m-terraform-state"
    key    = "cs410/network.tfstate"
    region = "us-east-1"

  }
}

data "aws_ami" "amazon2_linux" {
  most_recent = true

  filter {
    name = "name"
    # values = ["amzn2-ami-hvm-*-x86_64-gp2"]
    # values = ["ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"]
    # values = ["ubuntu/images/hvm-ssd/ubuntu*-amd64-server-*"]
    values = ["Deep Learning Base AMI (Ubuntu 18.04) Version 36.1"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }


  owners = ["amazon"]
  # owners = ["099720109477"] # Canonical
}

resource "aws_spot_instance_request" "cheap_worker" {
  # spot settings
  spot_price           = var.spot_price
  spot_type            = "one-time"
  wait_for_fulfillment = true
  valid_until          = timeadd(timestamp(), "10m")

  ami                         = var.ami_id
  instance_type               = var.instance_type
  key_name                    = var.ec2_key
  vpc_security_group_ids      = [var.ec2_security_group]
  associate_public_ip_address = true
  # user_data                   = local.user_data
  iam_instance_profile = var.ec2_iam_profile
  availability_zone    = var.ec2_availability_zone


  root_block_device {
    delete_on_termination = true
    volume_type           = "gp2"
    volume_size           = 80
    tags                  = local.tags

  }

  tags = local.tags
}


resource "aws_ec2_tag" "ec2_tag" {
  for_each    = local.tags
  resource_id = aws_spot_instance_request.cheap_worker.spot_instance_id
  key         = each.key
  value       = each.value
}

data "aws_ebs_volume" "ebs_volume" {
  most_recent = true

  filter {
    name   = "attachment.instance-id"
    values = [aws_spot_instance_request.cheap_worker.spot_instance_id]
  }
}

resource "aws_ec2_tag" "volume_atg" {
  for_each    = local.tags
  resource_id = data.aws_ebs_volume.ebs_volume.id
  key         = each.key
  value       = each.value
}


# resource "aws_instance" "cheap_worker" {
#   # spot settings
#   # spot_price           = var.spot_price
#   # spot_type            = "one-time"
#   # wait_for_fulfillment = true
#   # valid_until          = timeadd(timestamp(), "10m")
#   # lifecycle {
#   #   ignore_changes = [associate_public_ip_address]
#   # }
#   ami                         = data.aws_ami.amazon2_linux.id
#   instance_type               = var.instance_type
#   key_name                    = var.ec2_key
#   vpc_security_group_ids      = [var.ec2_security_group]
#   associate_public_ip_address = true
#   user_data                   = local.user_data
#   iam_instance_profile        = var.ec2_iam_profile
#   availability_zone           = var.ec2_availability_zone


#   root_block_device {
#     delete_on_termination = true
#     volume_type           = "gp2"
#     volume_size           = 80
#     tags                  = local.tags

#   }

#   tags = local.tags
# }
