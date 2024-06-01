# Database
If you want to use a personal/local database, follow these steps:

* Setup MySQL ([tutorial](https://dev.mysql.com/doc/refman/8.3/en/installing.html))
* Create database via `mysql> CREATE DATABASE carps;`
    Select password as authentification.
    Per default, the database name is `carps`.
    It is set in `carps/container/py_experimenter.yaml`.

2. Add credential file at `carps/container/credentials.yaml`, e.g.
```yaml
CREDENTIALS:
  Database:
    user: root
    password: <password>
  Connection:
    Standard:
      server: localhost
```
3. Set flag not to use ssh server in `carps/container/py_experimenter.yaml` if you are on your local machine.
