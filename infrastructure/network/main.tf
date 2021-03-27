provider "aws" {
  region = "us-east-2"
}

resource "aws_security_group" "private_ssh" {
  name        = "allow_ssh"
  description = "Allow OpenSSH traffic"
  vpc_id      = var.vpc_id

  ingress {
    description = "SSH from local"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.my_ip]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }


}
