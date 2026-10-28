import os
import win32com.client

ppt_path = os.path.abspath('d:/Coding/kaggle/RSNA/Knee_Abnormality_Detection_Presentation.pptx')
output_folder = os.path.abspath('assets/slides')
os.makedirs(output_folder, exist_ok=True)

powerpoint = win32com.client.Dispatch('PowerPoint.Application')
# Open presentation in read-only mode, without window if possible
presentation = powerpoint.Presentations.Open(ppt_path, ReadOnly=True, Untitled=False, WithWindow=False)

for i, slide in enumerate(presentation.Slides):
    slide_file = os.path.join(output_folder, f"slide_{i+1}.png")
    # 1920x1080 resolution
    slide.Export(slide_file, "PNG", 1920, 1080)
    print(f"Exported Slide {i+1} -> {slide_file}")

presentation.Close()
powerpoint.Quit()
print("All slides exported successfully.")
