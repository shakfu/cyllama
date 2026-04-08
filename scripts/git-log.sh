#!/usr/bin/env sh


echo "-------------------------------------------------------------"
echo "Changes over 1 year"
echo
git log --format=format: --name-only --since="1 year ago" \
	| sort | uniq -c | sort -nr | head -20

echo "-------------------------------------------------------------"
echo "Contributors"
echo
git shortlog -sn --no-merges

echo "-------------------------------------------------------------"
echo "log entries including fix/bug/broken"
echo
git log -i -E --grep="fix|bug|broken" --name-only --format='' \
	| sort | uniq -c | sort -nr | head -20

echo "-------------------------------------------------------------"
echo "Commits per month"
echo
git log --format='%ad' --date=format:'%Y-%m' | sort | uniq -c

echo "-------------------------------------------------------------"
echo "log entries with revert|hotfix|emergency|rollback"
echo
git log --oneline --since="1 year ago" |  grep -iE 'revert|hotfix|emergency|rollback'

