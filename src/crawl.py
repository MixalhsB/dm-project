from bs4 import BeautifulSoup
import re


def main():
    # crawl conditions:
    with open('values/conditions.html', 'r', encoding='utf-8') as f:
        sections = f.read().split('<a href=\u0022https://www.nhsinform.scot/illnesses-and-conditions/a-to-z/')[1:]

    conditions_list = []
    for sec in sections:
        conditions_list.append(re.findall(r'<h2.*?>(.+?)</h2>', sec.replace('\n', ' '))[0].strip())

    with open('test/UNMODIFIED_conditions.txt', 'w', encoding='utf-8') as f:
        for cond in conditions_list:
            f.write(cond + '\n')

    print('--> created UNMODIFIED_conditions.txt file.')


    # crawl therapies:
    with open('values/therapies.html', 'r', encoding='utf-8') as f:
        core = f.read().split('<p>This is a list of types of medical')[1].split('<h2><span class="mw-headline" id="See_also">See also')[0]

    sections = core.split('<li>')[1:]

    therapies_list = []
    for sec in sections:
        therapies_list.append(BeautifulSoup(re.findall(r'^(.+?)</li>', sec.replace('\n', ' '))[0].strip(), 'lxml').get_text())

    with open('test/UNMODIFIED_therapies.txt', 'w', encoding='utf-8') as f:
        for th in therapies_list:
            f.write(th + '\n')

    print('--> created UNMODIFIED_therapies.txt file.')


if __name__ == '__main__':
    main()
