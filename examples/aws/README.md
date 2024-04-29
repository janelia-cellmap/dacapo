You can work locally using S3 data by setting the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables. You can also set the `AWS_REGION` environment variable to specify the region to use. If you are using a profile, you can set the `AWS_PROFILE` environment variable to specify the profile to use.

```bash
aws configure
```

In order to store checkpoints and experiments data in S3, you need to modify `dacapo.yaml` to include the following:

```yaml
runs_base_dir: "s3://dacapotest"
```

For configs and stats, you can save them locally or s3 by setting `type: files` or for mongodb by setting `type: mongo` in the `dacapo.yaml` file.

