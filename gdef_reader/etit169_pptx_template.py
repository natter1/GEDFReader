# todo: move to python-pptx-interface
from datetime import datetime

import pkg_resources
from pptx_tools.templates import TemplateExample


# ETIT_16-9.pptx is not part of the repository due to legal restrictions!
class TemplateETIT169(TemplateExample):
    """
    Class handling ETIT 16:9 template. The needed file ETIT_16-9.pptx is not part of the repository
    due to legal restrictions. You can specify a path to the file (or a similar template pptx) via
    TemplateETIT169("..\\path\\to\\template\\my_template.pptx").
    """
    TEMPLATE_FILE = pkg_resources.resource_filename('pptx_tools', 'resources/ETIT_16-9.pptx')

    def __init__(self, template_file=None):
        if template_file:
            self.TEMPLATE_FILE = template_file

        super().__init__()
        date_time = datetime.now().strftime("%d %B, %Y")
        self.set_author("Nathanael Jöhrmann", city="Chemnitz", date=date_time)
        self.set_website("https://www.tu-chemnitz.de/etit/wetel/")

    def set_confidential(self, flag: bool = True):
        # todo: add confidential marker in master layout when flag == True
        pass
