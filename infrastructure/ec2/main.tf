provider "aws" {
  region = "us-east-2"
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
    values = ["ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }


  # owners = ["amazon"]
  owners = ["099720109477"] # Canonical
}

resource "aws_spot_instance_request" "cheap_worker" {
  # spot settings
  spot_price           = var.spot_price
  spot_type            = "one-time"
  wait_for_fulfillment = true
  valid_until          = timeadd(timestamp(), "10m")

  ami                         = data.aws_ami.amazon2_linux.id
  instance_type               = var.instance_type
  key_name                    = var.ec2_key
  vpc_security_group_ids      = [data.terraform_remote_state.network.outputs.security_group]
  associate_public_ip_address = true
  user_data                   = file("${path.module}/userdata.yaml")
  iam_instance_profile        = var.ec2_iam_profile


  root_block_device {
    delete_on_termination = true
    volume_type           = "gp2"
    volume_size           = 50

  }

}