variable "account_id" {
  type = string
}
variable "region" {
  type    = string
  default = "eu-west-2"
}
variable "project" {
  type    = string
  default = "qps"
}
variable "environment" {
  type    = string
  default = "test"
}
variable "source_key" {
  type    = string
  default = "sources/pending.zip"
}
variable "expires_at" {
  type = string
}
variable "source_sha" {
  type = string
}
