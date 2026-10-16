variable "prefix" {
  type = string
}
variable "vpc_id" {
  type = string
}
variable "subnet_ids" {
  type = list(string)
}
variable "client_security_group_ids" {
  type = map(string)
}
resource "aws_db_subnet_group" "main" {
  name       = "${var.prefix}-db"
  subnet_ids = var.subnet_ids
}
resource "aws_security_group" "database" {
  name        = "${var.prefix}-database"
  vpc_id      = var.vpc_id
  description = "Private PostgreSQL access from project workloads"
}
resource "aws_vpc_security_group_ingress_rule" "clients" {
  for_each                     = var.client_security_group_ids
  security_group_id            = aws_security_group.database.id
  referenced_security_group_id = each.value
  ip_protocol                  = "tcp"
  from_port                    = 5432
  to_port                      = 5432
}
resource "aws_db_instance" "main" {
  identifier                  = "${var.prefix}-db"
  engine                      = "postgres"
  engine_version              = "18.6"
  instance_class              = "db.t4g.micro"
  allocated_storage           = 20
  storage_type                = "gp3"
  storage_encrypted           = true
  db_name                     = "qps"
  username                    = "qpsadmin"
  manage_master_user_password = true
  publicly_accessible         = false
  multi_az                    = false
  db_subnet_group_name        = aws_db_subnet_group.main.name
  vpc_security_group_ids      = [aws_security_group.database.id]
  backup_retention_period     = 1
  delete_automated_backups    = true
  skip_final_snapshot         = true
  deletion_protection         = false
  auto_minor_version_upgrade  = false
  apply_immediately           = true
}
output "host" {
  value = aws_db_instance.main.address
}
output "port" {
  value = aws_db_instance.main.port
}
output "master_secret_arn" {
  value = aws_db_instance.main.master_user_secret[0].secret_arn
}
