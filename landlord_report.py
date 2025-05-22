import os
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.platypus import PageBreak, KeepTogether, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
import matplotlib.pyplot as plt
import io
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Modify the plotting backend for Streamlit compatibility
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit


class LandlordReportGenerator:
    #Class for generating PDF reports for landlords based on rental property analysis

    def __init__(self, output_dir="reports", label_encoders = None):
        # """Initialize the report generator with an output directory"""
        self.output_dir = output_dir
        self.label_encoders = label_encoders or {}

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Set up document styles
        self.styles = getSampleStyleSheet()

        # Create custom styles without overriding existing ones
        self._create_styles()

    def _decode_category(self, code, label_encoder):
        # """
        # Decode a category value using a label encoder

        # Parameters:
        # code: The encoded value (int/float)
        # label_encoder: The scikit-learn LabelEncoder that was used for encoding

        # Returns:
        # str: The decoded original string value
        # """
        if not hasattr(label_encoder, 'classes_'):
            return str(code)  # No classes to decode with

        try:
            # Convert to int if it's a float but represents an integer
            if isinstance(code, float) and code.is_integer():
                code = int(code)

            # Check if the code is a valid index
            if isinstance(code, int) and 0 <= code < len(label_encoder.classes_):
                return label_encoder.classes_[code]
        except:
            pass

        return str(code)  # Fall back to string representation

    def _decode_value(self, value, category):
        # """
        # Decode a category value using the label encoder.

        # This assumes that the encoded values are indices into the
        # label_encoder.classes_ array, which contains the original text values.

        # Parameters:
        # value: The encoded value (typically a numpy.int64)
        # category: Category name ('society', 'locality', etc.)

        # Returns:
        # str: The decoded text value
        # """
        # If already a string, return as is
        if isinstance(value, str):
            return value

        # Check if we have a label encoder for this category
        if category not in self.label_encoders:
            return str(value)

        # Get the encoder
        encoder = self.label_encoders[category]

        # Check if encoder has classes_
        if not hasattr(encoder, 'classes_'):
            return str(value)

        # Try to use the value as an index into the classes array
        try:
            # Convert numpy.int64 to Python int for indexing
            if hasattr(value, 'item'):
                index = value.item()  # For numpy.int64
            else:
                index = int(value)  # For regular integers or floats

            # Check if the index is valid
            if 0 <= index < len(encoder.classes_):
                return encoder.classes_[index]
        except:
            pass

        # Fall back to string representation
        return str(value)


    def _create_styles(self):
        # """Create custom styles for the report"""
        # Report title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=20,
            alignment=TA_CENTER,
            spaceAfter=15,
            textColor=colors.HexColor('#333333')  # Propico Purple
        ))

        # Section subtitle style
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            alignment=TA_LEFT,
            spaceAfter=10,
            textColor=colors.HexColor('#4A23AD')  # Propico Purple
        ))

        # Subsection heading style
        self.styles.add(ParagraphStyle(
            name='ReportHeading3',
            parent=self.styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#4A23AD'),  # Propico Purple - Light
            spaceAfter=8
        ))

        # Basic text style
        self.styles.add(ParagraphStyle(
            name='ReportBody',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            spaceAfter=6
        ))


        self.styles.add(ParagraphStyle(
            name='ReportBodyColor',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#4A23AD'),  # Propico Purple - Light
            leading=14,
            spaceAfter=6
        ))



        # Bold text style
        self.styles.add(ParagraphStyle(
            name='ReportBodyBold',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            spaceAfter=6,
            fontName='Helvetica-Bold'
        ))

        # Note/caption style
        self.styles.add(ParagraphStyle(
            name='ReportNote',
            parent=self.styles['Italic'],
            fontSize=9,
            textColor=colors.HexColor('#666666'),
            fontName='Helvetica'
        ))

        # Highlighted value style
        self.styles.add(ParagraphStyle(
            name='ReportHighlight',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#F26419'),  # Orange
            fontName='Helvetica-Bold',
            spaceAfter=6
        ))

    def generate_report(self, report_data, property_id=None):
        # """
        # Generate a PDF report based on the provided landlord report data

        # Parameters:
        # report_data: The dictionary returned by generate_landlord_report
        # property_id: Optional identifier for the property (for filename)

        # Returns:
        # str: Path to the generated PDF file
        # """
        # Set up filename
        if property_id is None:
            property_id = report_data.get('property_id', 'Unknown')

        date_str = datetime.now().strftime('%Y-%m-%d')
        filename = f"Rental_Analysis_{property_id}_{date_str}.pdf"
        filepath = os.path.join(self.output_dir, filename)

        # Create the document
        doc = SimpleDocTemplate(
            filepath,
            pagesize=A4,
            leftMargin=0.75*inch,
            rightMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )

        # Container for report elements
        elements = []

        # Build the report
        self._add_header(elements, report_data)
        self._add_property_details(elements, report_data)
        self._add_rent_estimate(elements, report_data)
        self._add_market_position(elements, report_data)
        self._add_comparables_analysis(elements, report_data)
        self._add_visualizations(elements, report_data)
        self._add_summary(elements, report_data)

        # Generate the PDF
        doc.build(elements)
        print(f"PDF report generated: {filepath}")
        return filepath

    def _add_header(self, elements, report_data):
        # """Add the report header section"""
        # Report title
        elements.append(Paragraph("Rental Market Analysis Report", self.styles['ReportTitle']))

        # Property identification
        property_id = report_data.get('property_id', 'Unknown')
        location = report_data.get('property_details', {}).get('location', {})
        locality_value = location.get('locality', 'Unknown Area')
        society_value = location.get('society', 'Unknown Society')
        # Format date
        # report_date = report_data.get('report_date', datetime.datetime.now().strftime('%Y-%m-%d'))
        report_date = '2025-05-15'

        # Decode society if encoded
        society_display = self._decode_value(society_value, 'society')
        locality_display = self._decode_value(locality_value, 'locality')

        subtitle_text = f"{society_display}, {locality_display}"
        # elements.append(Paragraph(subtitle_text, self.styles['ReportSubtitle']))

        # Date and reference
        date_text = f"Report Generated: {report_date}"
        # elements.append(Paragraph(date_text, self.styles['ReportNote']))

        # ref_text = f"Property ID: {property_id}"
        # elements.append(Paragraph(ref_text, self.styles['ReportNote']))

        # elements.append(Spacer(1, 0.1*inch))

        # Add horizontal line
        elements.append(Table([['']], colWidths=[6*inch, 0.1*inch], rowHeights=[0.05*inch],
                          style=TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), colors.HexColor('#FF5A6F')),
        ('GRID', (0, 0), (-1, -1), 0, colors.white),  # No grid lines
        ('ROUNDEDCORNERS', [2, 2, 2, 2]),  # Add rounded corners (if supported)
    ]),
                          hAlign='LEFT'
                              ))
        # Add negative space to move content up
        elements.append(Spacer(1, -0.1*inch))  # Negative space (adjust value as needed)

        # elements.append(Spacer(1, 0.2*inch))

    def _add_property_details(self, elements, report_data):
        # """Add the property details section"""
        # Section header
        elements.append(Paragraph("Property Details", self.styles['ReportSubtitle']))

        # Get property details
        details = report_data.get('property_details', {})
        physical = details.get('physical', {})
        building = details.get('building', {})
        condition = details.get('condition', {})
        pricing = details.get('pricing', {})
        location = report_data.get('property_details', {}).get('location', {})
        locality_value = location.get('locality', 'Unknown Area')
        society_value = location.get('society', 'Unknown Society')
        society_display = self._decode_value(society_value, 'society')
        locality_display = self._decode_value(locality_value, 'locality')

        # Create a table for property details (2 columns: label and value)
        data = []

        # Locality Row
        data.append(["Locality", locality_display])

        # Society Row
        data.append(["Society", society_display])

        # Configuration row
        bedrooms = physical.get('bedrooms', 0)
        bathrooms = physical.get('bathrooms', 0)
        config_text = f"{bedrooms} BHK, {bathrooms} Bathrooms"
        data.append(["Configuration", config_text])

        # Area row
        area = physical.get('builtup_area', 0)
        data.append(["Built-up Area", f"{area:,} sq.ft."])

        # Floor row
        floor = building.get('floor', 0)
        total_floors = building.get('total_floors', 0)
        data.append(["Floor", f"{floor} out of {total_floors}"])

        # Furnishing row
        furnishing_value = condition.get('furnishing', 'Unknown')
        furnishing_display = self._decode_value(furnishing_value, 'furnishing')
        data.append(["Furnishing", furnishing_display])

        # Rent details
        total_rent = pricing.get('total_rent', 0)
        rent_per_sqft = pricing.get('rent_per_sqft', 0)
        data.append(["Monthly Rent", f"Rs. {total_rent:,}"])
        data.append(["Rent per sq.ft.", f"Rs. {rent_per_sqft:.2f}"])

        # Create the table
        property_table = Table(data, colWidths=[2*inch, 3.5*inch])
        property_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#EAE6F5')),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#4A23AD')),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('TEXTCOLOR', (1, 6), (1, 7), colors.HexColor('#FF5A6F')),
            ('FONTNAME', (1, 6), (1, 7), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('BOX', (0, 0), (-1, -1), 1, colors.lightgrey),
            ('BACKGROUND', (1, 1), (1, 1), colors.whitesmoke),
            ('BACKGROUND', (1, 3), (1, 3), colors.whitesmoke),
            ('BACKGROUND', (1, 5), (1, 5), colors.whitesmoke),
            ('BACKGROUND', (1, 7), (1, 7), colors.whitesmoke),

        ]))

        elements.append(property_table)
        elements.append(Spacer(1, 0.1*inch))



    def _add_rent_estimate(self, elements, report_data):
        # """Add rent estimate section with range visualization"""
        # Section header
        elements.append(Paragraph("Rent Estimate", self.styles['ReportSubtitle']))

        # Get rent estimates data from report
        rent_estimates = report_data.get('rent_estimates', {})

        # If no rent estimates, show a message and return
        if not rent_estimates:
            elements.append(Paragraph("Rent estimate data not available.", self.styles['ReportBody']))
            return

        # Get values for visualization
        combined_estimate = rent_estimates.get('combined_estimate', 0)
        lower_bound = rent_estimates.get('lower_bound', 0)
        upper_bound = rent_estimates.get('upper_bound', 0)

        # Create rent estimate visualization
        elements.append(self._create_rent_estimate_visual(combined_estimate, lower_bound, upper_bound))
        elements.append(Spacer(1, 0.1*inch))

        # Add explanation text
        estimate_text = f"Based on our analysis, we estimate the optimal rent for your property to be " \
                      f"<font color='#FF5A6F'><b>Rs. {combined_estimate:,.0f}</b></font> per month. " \
                      f"This estimate has a range of Rs. {lower_bound:,.0f} to Rs. {upper_bound:,.0f}, " \
                      f"reflecting market conditions and property features."
        elements.append(Paragraph(estimate_text, self.styles['ReportBody']))

        # Add methodology note with details on the models used
        model_a = rent_estimates.get('model_a', 0)
        model_b = rent_estimates.get('model_b', 0)
        sqft_method = rent_estimates.get('sqft_method', 0)

        # Add explanation
        note_text = "This estimate combines machine learning predictions with comparable property analysis. " \
        "The lower bound represents a price point for faster occupancy, while the upper bound " \
                  "indicates the maximum market potential for your property."
        elements.append(Paragraph(note_text, self.styles['ReportNote']))

        elements.append(Spacer(1, 0.05*inch))

    def _create_rent_estimate_visual(self, estimated_rent, lower_bound, upper_bound):
        # """Create a visual representation of the rent estimate range"""
        # Calculate table data for visual representation
        data = [
            [f"Rs. {lower_bound:,.0f}", f"Rs. {estimated_rent:,.0f}", f"Rs. {upper_bound:,.0f}"],
            # ["", "/mo.", ""],
            ["", "", ""],  # Row for colored bar
            ["Rents Faster", "", "Rents Slower"]
        ]

        # Create table
        table = Table(data, colWidths=[1.8*inch, 1.8*inch, 1.8*inch])

        # Style the table
        table.setStyle(TableStyle([
            # Center alignment for all cells
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            # ('VALIGN', (0, 0), (2, 0), 'BOTTOM'),
            # ('VALIGN', (0, 1), (2, 2), 'MIDDLE'),


            # Format the estimate (center cell) with larger font
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica'),
            ('FONTSIZE', (1, 0), (1, 0), 12),
            ('TEXTCOLOR', (1, 0), (1, 0), colors.HexColor('#FF5A6F')),  # Propico

            # Format the bounds (left and right cells)
            ('FONTNAME', (0, 0), (0, 0), 'Helvetica'),
            ('FONTNAME', (2, 0), (2, 0), 'Helvetica'),
            ('FONTSIZE', (0, 0), (0, 0), 12),
            ('FONTSIZE', (2, 0), (2, 0), 12),
            ('TEXTCOLOR', (0, 0), (0, 0), colors.HexColor('#395B74')),  # Dark blue
            ('TEXTCOLOR', (2, 0), (2, 0), colors.HexColor('#395B74')),  # Dark blue

            # # Format the "/mo." text
            # ('FONTNAME', (1, 1), (1, 1), 'Helvetica'),
            # ('FONTSIZE', (1, 1), (1, 1), 10),
            # ('TEXTCOLOR', (1, 1), (1, 1), colors.HexColor('#395B74')),  # Dark blue

            # Color bar (gradient simulation with three colors)
            ('BACKGROUND', (0, 1), (0, 1), colors.HexColor('#F05454')),  # Left red
            ('BACKGROUND', (1, 1), (1, 1), colors.HexColor('#81B29A')),  # Middle green
            ('BACKGROUND', (2, 1), (2, 1), colors.HexColor('#F05454')),  # Right red
            ('LINEABOVE', (0, 1), (2, 1), 1, colors.white),
            ('LINEBELOW', (0, 1), (2, 1), 1, colors.white),

            # Format "Rents Faster" and "Rents Slower" text
            ('FONTNAME', (0, 2), (0, 2), 'Helvetica'),
            ('FONTNAME', (2, 2), (2, 2), 'Helvetica'),
            ('FONTSIZE', (0, 2), (0, 2), 10),
            ('FONTSIZE', (2, 2), (2, 2), 10),
            ('TEXTCOLOR', (0, 2), (0, 2), colors.HexColor('#395B74')),  # Dark blue
            ('TEXTCOLOR', (2, 2), (2, 2), colors.HexColor('#395B74')),  # Dark blue
        ]))

        return table

    def _add_market_position(self, elements, report_data):
        # """Add the market position section"""
        # Section header
        elements.append(Paragraph("Market Position Analysis", self.styles['ReportSubtitle']))

        # Get market position data
        market_position = report_data.get('market_position', {})
        position_category = market_position.get('position_category', 'Unknown')
        primary_group = market_position.get('primary_comparison_group', 'overall_market')
        premium_discount = market_position.get('premium_discount', {})
        percentile_ranks = market_position.get('percentile_ranks', {})

        # Format the primary comparison group name
        primary_group_display = primary_group.replace('_', ' ').title()

        # Position statement
        position_text = f"Your property is positioned in the <font color='#FF5A6F'><b>{position_category}</b></font> segment, " \
                        f"based on comparison with {primary_group_display} properties."
        elements.append(Paragraph(position_text, self.styles['ReportBody']))

        # Add Percentile Rankings
        elements.append(Paragraph("Percentile Rankings", self.styles['ReportHeading3']))

        percentile_data = []
        percentile_data.append(["Comparison Group", "Rent Percentile", "Rent/sq-ft Percentile"])

        # Add percentiles for each comparison group
        actual_data_rows = 0
        for group_name, values in percentile_ranks.items():
            if group_name != 'bedroom_type':  # Skip this for cleaner report
                group_display = group_name.replace('_', ' ').title()
                rent_percentile = values.get('rent', 0)
                rent_sqft_percentile = values.get('rent_sqft', 0)

                # Format percentiles with one decimal place
                percentile_data.append([
                    group_display,
                    f"{rent_percentile:.1f}th",
                    f"{rent_sqft_percentile:.1f}th"
                ])
                actual_data_rows+=1

        if actual_data_rows >0:
            # Create percentile table
            percentile_table = Table(percentile_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
            total_rows = len(percentile_data)  # includes header
            style_commands = [
                # Header row styling
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A23AD')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
                # Data rows styling
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),  # Make first column bold
                # Cell padding for better spacing
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                # Borders - more subtle
                ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ]
    
            # ONLY add highlighting if we have enough rows (at least 4 total = header + 3 data)
            if total_rows >= 4:
                style_commands.extend([
                    ('TEXTCOLOR', (1, 3), (2, 3), colors.HexColor('#FF5A6F')),
                    ('FONTNAME', (1, 3), (2, 3), 'Helvetica-Bold'),
                ])
    
            # ONLY add alternating row colors if we have enough rows
            if total_rows >= 3:  # header + 2 data rows
                style_commands.append(('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#EAE6F5')))
            if total_rows >= 5:  # header + 4 data rows
                style_commands.append(('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#EAE6F5')))
            if total_rows >= 7:  # header + 6 data rows
                style_commands.append(('BACKGROUND', (0, 6), (-1, 6), colors.HexColor('#EAE6F5')))
    
            percentile_table.setStyle(TableStyle(style_commands))
            elements.append(percentile_table)

        else:
            elements.append(Paragraph("Percentile ranking data not available.", self.styles['ReportBody']))

        elements.append(Spacer(1, 0.2*inch))

        progress_title = Paragraph("Your Market Position", self.styles['ReportBodyColor'])
        elements.append(progress_title)
        elements.append(Spacer(1, 0.1*inch))

        # Get the appropriate percentile for the progress bar
        # Use the 'overall_market' percentile or fallback to the first available value
        bar_percentile = 0
        if 'same_society_same_bhk' in percentile_ranks:
            bar_percentile = round(percentile_ranks['same_society_same_bhk'].get('rent_sqft', 0), 1)
        # If not available, fall back to overall_market
        elif 'overall_market' in percentile_ranks:
            bar_percentile = round(percentile_ranks['overall_market'].get('rent_sqft', 0), 1)
        # Last resort - use first available group
        elif percentile_ranks:
            first_group = next(iter(percentile_ranks.values()))
            bar_percentile = round(first_group.get('rent_sqft', 0), 1)

        # Create a simple progress bar using a table
        primary_color = colors.HexColor('#4A23AD')  # Purple
        accent_color = colors.HexColor('#FF5A6F')   # Coral pink
        bg_color = colors.HexColor('#E0E0E0')       # Light gray background

        # Calculate filled width based on percentage
        percent = bar_percentile
        # Reserve space for percentage display at the right
        percent_width = 0.7*inch

        # Calculate bar width (total width minus percentage display width)
        total_width = 5.5*inch
        bar_width = total_width - percent_width

        # Calculate filled width based on percentage
        filled_width = (percent / 100) * bar_width
        empty_width = bar_width - filled_width

        # Create percentage text with proper styling
        percent_text = Paragraph(f"{percent}%",
                              ParagraphStyle(
                                  'PercentText',
                                  fontName='Helvetica-Bold',
                                  fontSize=12,
                                  textColor=primary_color,
                                  alignment=1  # Center aligned
                              ))

        # Create a row with three cells: filled portion, empty portion, and percentage
        if empty_width > 0:
            # If bar is not 100% filled
            bar_data = [["", "", percent_text]]
            bar_table = Table(bar_data,
                            colWidths=[filled_width, empty_width, percent_width],
                            rowHeights=[0.3*inch])

            # Style the progress bar
            bar_table.setStyle(TableStyle([
                # Fill the first cell with accent color
                ('BACKGROUND', (0, 0), (0, 0), accent_color),
                # Fill the second cell with background color
                ('BACKGROUND', (1, 0), (1, 0), bg_color),
                # Make percentage cell transparent
                ('BACKGROUND', (2, 0), (2, 0), colors.transparent),
                # Center align the percentage
                ('ALIGN', (2, 0), (2, 0), 'CENTER'),
                ('VALIGN', (2, 0), (2, 0), 'MIDDLE'),
                # Remove all grid lines
                ('GRID', (0, 0), (-1, -1), 0, colors.white),
                # Add a subtle border around just the bar portion (not the percentage)
                ('BOX', (0, 0), (1, 0), 0.5, colors.lightgrey),
            ]))
        else:
            # If bar is 100% filled
            bar_data = [["", percent_text]]
            bar_table = Table(bar_data,
                            colWidths=[bar_width, percent_width],
                            rowHeights=[0.3*inch])

            # Style the progress bar
            bar_table.setStyle(TableStyle([
                # Fill the first cell with accent color
                ('BACKGROUND', (0, 0), (0, 0), accent_color),
                # Make percentage cell transparent
                ('BACKGROUND', (1, 0), (1, 0), colors.transparent),
                # Center align the percentage
                ('ALIGN', (1, 0), (1, 0), 'CENTER'),
                ('VALIGN', (1, 0), (1, 0), 'MIDDLE'),
                # Remove all grid lines
                ('GRID', (0, 0), (-1, -1), 0, colors.white),
                # Add a subtle border around just the bar portion (not the percentage)
                ('BOX', (0, 0), (0, 0), 0.5, colors.lightgrey),
            ]))

        # Add rounded corners if your ReportLab version supports it
        try:
            if empty_width > 0:
                bar_table.setStyle(TableStyle([('ROUNDEDCORNERS', [5, 5, 5, 5], (0, 0), (1, 0))]))
            else:
                bar_table.setStyle(TableStyle([('ROUNDEDCORNERS', [5, 5, 5, 5], (0, 0), (0, 0))]))
        except:
            # If ROUNDEDCORNERS is not supported, continue without it
            pass

        # Add the progress bar to the elements
        elements.append(bar_table)

        # Add some padding at the bottom
        elements.append(Spacer(1, 0.5*inch))

        # Add Premium/Discount Analysis
        elements.append(Paragraph("Premium/Discount Analysis", self.styles['ReportHeading3']))

        # Get premium/discount values for important comparison groups
        society_premium = premium_discount.get('same_society_same_bhk_avg',
                                              premium_discount.get('society_avg', 0))
        locality_premium = premium_discount.get('same_locality_same_bhk_avg',
                                               premium_discount.get('locality_avg', 0))
        overall_premium = premium_discount.get('overall_market_avg',
                                              premium_discount.get('locality_avg', 0))
        comparables_premium = premium_discount.get('comparables_avg', overall_premium)

        # Create premium/discount table
        premium_data = []
        premium_data.append(["Comparison Group", "Premium/Discount"])
        # Function to create formatted paragraph with color
        def create_premium_paragraph(value):
            status = "PREMIUM" if value > 0 else "DISCOUNT"
            color = colors.HexColor('#2E8B57') if value > 0 else colors.HexColor('#D32F2F')
            value_text = f"{abs(value):.1f}% {status}"

            # Create a paragraph with direct color styling (not HTML)
            style = ParagraphStyle(
                name='PremiumStyle',
                parent=self.styles['ReportBody'],
                alignment=TA_CENTER,
                textColor=color
            )
            return Paragraph(value_text, style)

        # Add rows with formatted premium values using Paragraph objects
        premium_data_rows = 0
        for label, value in [
            ("Same Society & BHK", society_premium),
            ("Same Locality & BHK", locality_premium),
            ("Overall Market", overall_premium),
            ("Primary Comparables", comparables_premium)
        ]:
            # Use Paragraph for the label as well to keep consistent styling
            label_para = Paragraph(label, self.styles['ReportBody'])
            value_para = create_premium_paragraph(value)
            premium_data.append([label_para, value_para])
            premium_data_rows += 1

        if premium_data_rows > 0:

            # Create premium/discount table
            primary_color = colors.HexColor('#4A23AD')  # Purple
            accent_color = colors.HexColor('#FF5A6F')   # Coral
            premium_color = colors.HexColor('#2E8B57')  # Green for premium
            discount_color = colors.HexColor('#D32F2F')  # Red for discount
            header_color = colors.HexColor('#455A64')   # Dark slate for header
            light_purple = colors.HexColor('#EAE6F5')   # Light purple for alternating rows
    
            premium_table = Table(premium_data, colWidths=[3*inch, 3*inch])
            # SAFE STYLING for premium table
            premium_total_rows = len(premium_data)
            premium_style_commands = [
                # Header row styling
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A23AD')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
                # Data rows styling
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
                # Cell padding for better spacing
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                # Borders - more subtle
                ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ]
    
            # ONLY add alternating backgrounds if enough rows
            if premium_total_rows >= 3:
                premium_style_commands.append(('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#EAE6F5')))
            if premium_total_rows >= 5:
                premium_style_commands.append(('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#EAE6F5')))
    
            premium_table.setStyle(TableStyle(premium_style_commands))
            elements.append(premium_table)

        else:
            elements.append(Paragraph("Premium/discount analysis not available.", self.styles['ReportBody']))

        elements.append(Spacer(1, 0.1*inch))

    def _add_comparables_analysis(self, elements, report_data):
        """Add the comparables analysis section with safe table handling for few/no comparables"""
        # Section header
        elements.append(Paragraph("Comparable Properties Analysis", self.styles['ReportSubtitle']))
    
        # Get comparables data
        comparables = report_data.get('comparables', {})
        tiered_analysis = comparables.get('tiered_analysis', {})
    
        elements.append(Paragraph("The following analysis shows how your property compares to different segments of the market:", self.styles['ReportBody']))
    
        # Create comparable properties table
        comp_data = []
        comp_data.append(["Comparable Group", "Properties", "Avg. Rent", "Rent Range", "Your Position"])
    
        # Add data for each tier with sufficient data - ONLY ADD ROWS THAT HAVE DATA
        rows_added = 0
        for tier_name, tier_data in tiered_analysis.items():
            if tier_data.get('available', False) and tier_data.get('count', 0) >= 3:
                tier_display_name = tier_name.replace('_', ' ').title()
                count = tier_data.get('count', 0)
                avg_rent = tier_data.get('avg_rent', 0)
                min_rent = tier_data.get('min_rent', 0)
                max_rent = tier_data.get('max_rent', 0)
                premium = tier_data.get('premium_discount', 0)
    
                # Format position text
                if premium > 0:
                    position_text = f"{abs(premium):.1f}% above avg."
                else:
                    position_text = f"{abs(premium):.1f}% below avg."
    
                # Add row to table
                comp_data.append([
                    tier_display_name,
                    str(count),
                    f"Rs. {avg_rent:,.0f}",
                    f"Rs. {min_rent:,.0f} - Rs. {max_rent:,.0f}",
                    position_text
                ])
                rows_added += 1
    
        # CHECK IF WE HAVE ENOUGH DATA FOR A TABLE
        if rows_added == 0:
            # No comparable data available
            elements.append(Paragraph("Insufficient comparable properties found for detailed analysis. This may be due to:", self.styles['ReportBody']))
            elements.append(Paragraph("• Limited properties in the same society", self.styles['ReportBody']))
            elements.append(Paragraph("• Few properties with the same configuration in the area", self.styles['ReportBody']))
            elements.append(Paragraph("• Unique property characteristics", self.styles['ReportBody']))
            elements.append(Spacer(1, 0.2*inch))
            return
    
        # Create the comparables table ONLY if we have data
        comp_table = Table(comp_data, colWidths=[2*inch, 0.8*inch, 1*inch, 1.7*inch, 1.4*inch])
    
        # Calculate actual table dimensions
        actual_rows = len(comp_data)  # This includes header + data rows
        actual_cols = 5
    
        # SAFE STYLING - only apply styles to rows that actually exist
        style_commands = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A23AD')),  # Header row
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),  # Right-align data cells
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ]
    
        # ONLY add alternating row colors if we have enough rows
        if actual_rows >= 3:  # Header + at least 2 data rows
            style_commands.append(('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#EAE6F5')))
        if actual_rows >= 5:  # Header + at least 4 data rows
            style_commands.append(('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#EAE6F5')))
    
        comp_table.setStyle(TableStyle(style_commands))
        elements.append(comp_table)
        elements.append(Spacer(1, 0.2*inch))
    
        # Add a similar table for Rent per Square Foot
        elements.append(Paragraph("Comparison by Rent per Square Foot", self.styles['ReportHeading3']))
    
        psf_data = []
        psf_data.append(["Comparable Group", "Properties", "Avg. Rs./sq-ft", "Rs./sq-ft Range", "Your Position"])
    
        # Get property's rent per sqft for comparison
        property_details = report_data.get('property_details', {})
        pricing = property_details.get('pricing', {})
        rent_per_sqft = pricing.get('rent_per_sqft', 0)
    
        # Add data for each tier with sufficient data - AGAIN, ONLY ADD ROWS WITH DATA
        psf_rows_added = 0
        for tier_name, tier_data in tiered_analysis.items():
            if tier_data.get('available', False) and tier_data.get('count', 0) >= 3:
                tier_display_name = tier_name.replace('_', ' ').title()
                count = tier_data.get('count', 0)
    
                # Get rent per sqft metrics
                avg_rent_psf = tier_data.get('avg_rent_psf', 0)
                premium_psf = tier_data.get('premium_discount_psf', 0)
    
                # Calculate range if available in data or estimate
                if 'min_rent_psf' in tier_data and 'max_rent_psf' in tier_data:
                    min_rent_psf = tier_data.get('min_rent_psf', 0)
                    max_rent_psf = tier_data.get('max_rent_psf', 0)
                else:
                    # Estimate range based on total rent range and average area
                    min_rent = tier_data.get('min_rent', 0)
                    max_rent = tier_data.get('max_rent', 0)
                    avg_area = property_details.get('physical', {}).get('builtup_area', 1000)
                    
                    # Avoid division by zero
                    if avg_area > 0:
                        min_rent_psf = min_rent / avg_area
                        max_rent_psf = max_rent / avg_area
                    else:
                        min_rent_psf = 0
                        max_rent_psf = 0
    
                # Format position text
                if premium_psf > 0:
                    position_text = f"{abs(premium_psf):.1f}% above avg."
                else:
                    position_text = f"{abs(premium_psf):.1f}% below avg."
    
                # Add row to table
                psf_data.append([
                    tier_display_name,
                    str(count),
                    f"Rs. {avg_rent_psf:.2f}",
                    f"Rs. {min_rent_psf:.2f} - Rs. {max_rent_psf:.2f}",
                    position_text
                ])
                psf_rows_added += 1
    
        # CHECK AGAIN FOR PSF TABLE
        if psf_rows_added == 0:
            elements.append(Paragraph("Rent per square foot comparison not available due to insufficient comparable data.", self.styles['ReportBody']))
            elements.append(Spacer(1, 0.2*inch))
            return
    
        # Create the rent per sqft table
        psf_table = Table(psf_data, colWidths=[2*inch, 0.8*inch, 1*inch, 1.8*inch, 1.45*inch])
    
        # Calculate actual PSF table dimensions
        actual_psf_rows = len(psf_data)
        actual_psf_cols = 5
    
        # SAFE STYLING for PSF table
        psf_style_commands = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A23AD')),  # Header row
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),  # Right-align data cells
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ]
    
        # ONLY add alternating row colors if we have enough rows
        if actual_psf_rows >= 3:  # Header + at least 2 data rows
            psf_style_commands.append(('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#EAE6F5')))
        if actual_psf_rows >= 5:  # Header + at least 4 data rows
            psf_style_commands.append(('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#EAE6F5')))
    
        psf_table.setStyle(TableStyle(psf_style_commands))
        elements.append(psf_table)
        elements.append(Spacer(1, 0.2*inch))
    
        # Add insight about comparables - ONLY if we have data
        if psf_rows_added > 0:
            property_rent = pricing.get('total_rent', 0)
    
            # Get most relevant comparable group stats
            primary_group = report_data.get('market_position', {}).get('primary_comparison_group', 'same_locality_same_bhk')
            primary_tier = None
    
            for tier_name, tier_data in tiered_analysis.items():
                if tier_name == primary_group and tier_data.get('available', False):
                    primary_tier = tier_data
                    break
    
            if primary_tier:
                avg_rent = primary_tier.get('avg_rent', 0)
                premium = primary_tier.get('premium_discount', 0)
                avg_rent_psf = primary_tier.get('avg_rent_psf', 0)
                premium_psf = primary_tier.get('premium_discount_psf', 0)
    
                if premium > 10:
                    insight_text = f"Your property's rent of Rs. {property_rent:,} is significantly higher than the average " \
                                  f"of Rs. {avg_rent:,.0f} for comparable properties in {primary_group.replace('_', ' ')}. " \
                                  f"This may reflect superior features or positioning."
                elif premium > 0:
                    insight_text = f"Your property is priced moderately above comparable properties " \
                                  f"in {primary_group.replace('_', ' ')}, commanding a {premium:.1f}% premium."
                elif premium > -10:
                    insight_text = f"Your property is priced slightly below comparable properties " \
                                  f"in {primary_group.replace('_', ' ')}, which may help with faster occupancy."
                else:
                    insight_text = f"Your property is priced significantly below comparable properties " \
                                  f"in {primary_group.replace('_', ' ')}. There may be opportunity to " \
                                  f"increase rent in the future."
    
                elements.append(Paragraph(insight_text, self.styles['ReportBody']))
    
                # Add insight based on rent per sqft
                elements.append(Spacer(1, 0.05*inch))
    
                if premium_psf > 10:
                    psf_insight = f"On a per square foot basis (Rs. {rent_per_sqft:.2f}/sqft), your property commands a significant " \
                                 f"premium of {premium_psf:.1f}% over the average (Rs. {avg_rent_psf:.2f}/sqft) in this segment. " \
                                 f"This suggests excellent value relative to its size."
                elif premium_psf > 0:
                    psf_insight = f"Your property's per square foot rate (Rs.{rent_per_sqft:.2f}/sqft) is moderately higher than " \
                                 f"the average (Rs.{avg_rent_psf:.2f}/sqft) for comparable properties, indicating good pricing relative to size."
                elif premium_psf > -10:
                    psf_insight = f"Your property's per square foot rate (Rs.{rent_per_sqft:.2f}/sqft) is slightly below " \
                                 f"the average (Rs.{avg_rent_psf:.2f}/sqft) for comparable properties, suggesting fair value for its size."
                else:
                    psf_insight = f"Your property's per square foot rate (Rs.{rent_per_sqft:.2f}/sqft) is significantly lower than " \
                                 f"the average (Rs.{avg_rent_psf:.2f}/sqft) for comparable properties. This could indicate " \
                                 f"potential for a rent increase or reflect specific property characteristics."
    
                elements.append(Paragraph(psf_insight, self.styles['ReportBody']))
    
    # def _add_comparables_analysis(self, elements, report_data):
    #     # """Add the comparables analysis section"""
    #     # Section header
    #     elements.append(Paragraph("Comparable Properties Analysis", self.styles['ReportSubtitle']))

    #     # Get comparables data
    #     comparables = report_data.get('comparables', {})
    #     tiered_analysis = comparables.get('tiered_analysis', {})

    #     elements.append(Paragraph("The following analysis shows how your property compares to different segments of the market:", self.styles['ReportBody']))

    #     # Create comparable properties table
    #     comp_data = []
    #     comp_data.append(["Comparable Group", "Properties", "Avg. Rent", "Rent Range", "Your Position"])

    #     # Add data for each tier with sufficient data
    #     for tier_name, tier_data in tiered_analysis.items():
    #         if tier_data.get('available', False) and tier_data.get('count', 0) >= 3:
    #             tier_display_name = tier_name.replace('_', ' ').title()
    #             count = tier_data.get('count', 0)
    #             avg_rent = tier_data.get('avg_rent', 0)
    #             min_rent = tier_data.get('min_rent', 0)
    #             max_rent = tier_data.get('max_rent', 0)
    #             percentile = tier_data.get('percentile', 50)
    #             premium = tier_data.get('premium_discount', 0)

    #             # Format position text
    #             if premium > 0:
    #                 position_text = f"{abs(premium):.1f}% above avg."
    #             else:
    #                 position_text = f"{abs(premium):.1f}% below avg."

    #             # Add row to table
    #             comp_data.append([
    #                 tier_display_name,
    #                 str(count),
    #                 f"Rs. {avg_rent:,.0f}",
    #                 f"Rs. {min_rent:,.0f} - Rs. {max_rent:,.0f}",
    #                 position_text
    #             ])

    #     # Create the comparables table       
    #     comp_table = Table(comp_data, colWidths=[2*inch, 0.8*inch, 1*inch, 1.7*inch, 1.4*inch])
    #     comp_table.setStyle(TableStyle([
    #         ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A23AD')),  # Medium blue for header row
    #         ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    #         ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
    #         ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),  # Right-align data cells
    #         ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
    #         ('FONTNAME', (0, 1), (0, -1), 'Helvetica'),
    #         ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    #         ('TOPPADDING', (0, 0), (-1, -1), 6),
    #         ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
    #         # Alternating row colors for readability
    #         ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#EAE6F5')),
    #         ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#EAE6F5')),
    #     ]))

    #     elements.append(comp_table)
    #     elements.append(Spacer(1, 0.2*inch))

    #     # Add a similar table for Rent per Square Foot
    #     elements.append(Paragraph("Comparison by Rent per Square Foot", self.styles['ReportHeading3']))

    #     psf_data = []
    #     psf_data.append(["Comparable Group", "Properties", "Avg. Rs./sq-ft", "Rs./sq-ft Range", "Your Position"])

    #     # Get property's rent per sqft for comparison
    #     property_details = report_data.get('property_details', {})
    #     pricing = property_details.get('pricing', {})
    #     rent_per_sqft = pricing.get('rent_per_sqft', 0)

    #     # Add data for each tier with sufficient data
    #     for tier_name, tier_data in tiered_analysis.items():
    #         if tier_data.get('available', False) and tier_data.get('count', 0) >= 3:
    #             tier_display_name = tier_name.replace('_', ' ').title()
    #             count = tier_data.get('count', 0)

    #             # Get rent per sqft metrics
    #             avg_rent_psf = tier_data.get('avg_rent_psf', 0)

    #             # Get premium/discount based on rent per sqft
    #             premium_psf = tier_data.get('premium_discount_psf', 0)

    #             # Calculate range if available in data or estimate
    #             if 'min_rent_psf' in tier_data and 'max_rent_psf' in tier_data:
    #                 min_rent_psf = tier_data.get('min_rent_psf', 0)
    #                 max_rent_psf = tier_data.get('max_rent_psf', 0)
    #             else:
    #                 # Estimate range based on total rent range and average area
    #                 min_rent = tier_data.get('min_rent', 0)
    #                 max_rent = tier_data.get('max_rent', 0)
    #                 avg_area = property_details.get('physical', {}).get('builtup_area', 1000)

    #                 # Avoid division by zero
    #                 if avg_area > 0:
    #                     min_rent_psf = min_rent / avg_area
    #                     max_rent_psf = max_rent / avg_area
    #                 else:
    #                     min_rent_psf = 0
    #                     max_rent_psf = 0

    #             # Format position text
    #             if premium_psf > 0:
    #                 position_text = f"{abs(premium_psf):.1f}% above avg."
    #             else:
    #                 position_text = f"{abs(premium_psf):.1f}% below avg."

    #             # Add row to table
    #             psf_data.append([
    #                 tier_display_name,
    #                 str(count),
    #                 f"Rs. {avg_rent_psf:.2f}",
    #                 f"Rs. {min_rent_psf:.2f} - Rs. {max_rent_psf:.2f}",
    #                 position_text
    #             ])

    #     # Create the rent per sqft table
    #     psf_table = Table(psf_data, colWidths=[2*inch, 0.8*inch, 1*inch, 1.8*inch, 1.45*inch])
    #     psf_table.setStyle(TableStyle([
    #         ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A23AD')),  # Medium blue for header row
    #         ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    #         ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
    #         ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),  # Right-align data cells
    #         ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
    #         ('FONTNAME', (0, 1), (0, -1), 'Helvetica'),
    #         ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    #         ('TOPPADDING', (0, 0), (-1, -1), 6),
    #         ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
    #         # Alternating row colors for readability
    #         ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#EAE6F5')),
    #         ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#EAE6F5')),

    #     ]))

    #     elements.append(psf_table)
    #     elements.append(Spacer(1, 0.2*inch))


    #     # Add insight about comparables
    #     property_details = report_data.get('property_details', {})
    #     pricing = property_details.get('pricing', {})
    #     rent = pricing.get('total_rent', 0)

    #     # Get most relevant comparable group stats
    #     primary_group = report_data.get('market_position', {}).get('primary_comparison_group', 'same_locality_same_bhk')
    #     primary_tier = None
    #     for tier_name, tier_data in tiered_analysis.items():
    #         if tier_name == primary_group and tier_data.get('available', False):
    #             primary_tier = tier_data
    #             break

    #     if primary_tier:
    #         avg_rent = primary_tier.get('avg_rent', 0)
    #         premium = primary_tier.get('premium_discount', 0)
    #         # Get the per-sqft metrics from the SAME primary_tier
    #         avg_rent_psf = primary_tier.get('avg_rent_psf', 0)
    #         premium_psf = primary_tier.get('premium_discount_psf', 0)

    #         if premium > 10:
    #             insight_text = f"Your property's rent of Rs. {rent:,} is significantly higher than the average " \
    #                            f"of Rs. {avg_rent:,.0f} for comparable properties in {primary_group.replace('_', ' ')}. " \
    #                            f"This may reflect superior features or positioning."
    #         elif premium > 0:
    #             insight_text = f"Your property is priced moderately above comparable properties " \
    #                            f"in {primary_group.replace('_', ' ')}, commanding a {premium:.1f}% premium."
    #         elif premium > -10:
    #             insight_text = f"Your property is priced slightly below comparable properties " \
    #                            f"in {primary_group.replace('_', ' ')}, which may help with faster occupancy."
    #         else:
    #             insight_text = f"Your property is priced significantly below comparable properties " \
    #                            f"in {primary_group.replace('_', ' ')}. There may be opportunity to " \
    #                            f"increase rent in the future."

    #         elements.append(Paragraph(insight_text, self.styles['ReportBody']))

    #         # Add insight based on rent per sqft
    #         elements.append(Spacer(1, 0.05*inch))
    #         if premium_psf > 10:
    #             psf_insight = f"On a per square foot basis (Rs. {rent_per_sqft:.2f}/sqft), your property commands a significant " \
    #                         f"premium of {premium_psf:.1f}% over the average (Rs. {avg_rent_psf:.2f}/sqft) in this segment. " \
    #                         f"This suggests excellent value relative to its size."
    #         elif premium_psf > 0:
    #             psf_insight = f"Your property's per square foot rate (Rs.{rent_per_sqft:.2f}/sqft) is moderately higher than " \
    #                         f"the average (Rs.{avg_rent_psf:.2f}/sqft) for comparable properties, indicating good pricing relative to size."
    #         elif premium_psf > -10:
    #             psf_insight = f"Your property's per square foot rate (Rs.{rent_per_sqft:.2f}/sqft) is slightly below " \
    #                         f"the average (Rs.{avg_rent_psf:.2f}/sqft) for comparable properties, suggesting fair value for its size."
    #         else:
    #             psf_insight = f"Your property's per square foot rate (Rs.{rent_per_sqft:.2f}/sqft) is significantly lower than " \
    #                         f"the average (Rs.{avg_rent_psf:.2f}/sqft) for comparable properties. This could indicate " \
    #                         f"potential for a rent increase or reflect specific property characteristics."

            # elements.append(Paragraph(psf_insight, self.styles['ReportBody']))


        # elements.append(Spacer(1, 0.05*inch))

    def _add_visualizations(self, elements, report_data):
        # """Add visualization charts to the report"""
        # Section header
        elements.append(Paragraph("Market Visualizations", self.styles['ReportSubtitle']))

        # Get visualization generators
        visualization_generators = report_data.get('visualization_generators', {})
        visualizations = report_data.get('visualizations', {})

        # Function to render matplotlib figures to ReportLab images
        def fig_to_image(fig, width=6*inch, max_height=4.5*inch):
            # """Convert matplotlib figure to ReportLab Image"""
            # Get the figure's aspect ratio
            fig_width = fig.get_figwidth()
            fig_height = fig.get_figheight()
            aspect_ratio = fig_height / fig_width
            height = width * aspect_ratio

            # If height exceeds maximum, scale down width to maintain aspect ratio
            if height > max_height:
                width = max_height / aspect_ratio
                height = max_height

            # Save figure to buffer with appropriate DPI
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)

            # Create image with explicit width and height
            img = Image(buf, width=width, height=height)
            plt.close(fig)  # Close the figure to free memory
            return img

        # Try to add position chart
        elements.append(Paragraph("Your Property vs. Comparable Properties", self.styles['ReportHeading3']))
        position_chart = None

        if 'position_chart' in visualizations:
            position_chart = visualizations['position_chart']
        elif 'position_chart' in visualization_generators:
            try:
                position_chart = visualization_generators['position_chart']()
            except Exception as e:
                print(f"Error generating position chart: {e}")

        if position_chart:
            elements.append(fig_to_image(position_chart))
            elements.append(Paragraph("This chart shows how your property's rent (red line) compares to similar properties in the same market segment.", self.styles['ReportNote']))
        else:
            elements.append(Paragraph("Position chart not available", self.styles['ReportNote']))

        elements.append(Spacer(1, 0.2*inch))

        # Add feature radar chart
        elements.append(Paragraph("Property Feature Comparison", self.styles['ReportHeading3']))
        feature_radar = None

        if 'feature_radar' in visualizations:
            feature_radar = visualizations['feature_radar']
        elif 'feature_radar' in visualization_generators:
            try:
                feature_radar = visualization_generators['feature_radar']()
            except Exception as e:
                print(f"Error generating feature radar: {e}")

        if feature_radar:
            elements.append(fig_to_image(feature_radar))
            elements.append(Paragraph("This radar chart compares your property's key features to comparable properties. Areas where your property's line (red) extends further than others indicate competitive strengths.", self.styles['ReportNote']))
        else:
            elements.append(Paragraph("Feature radar chart not available", self.styles['ReportNote']))

        # elements.append(PageBreak())  # Start feature impact on new page

        # Add feature impact chart
        elements.append(Paragraph("Factors Affecting Rent Value", self.styles['ReportHeading3']))
        feature_impact = None

        if 'feature_impact' in visualizations:
            feature_impact = visualizations['feature_impact']
        elif 'feature_impact' in visualization_generators:
            try:
                feature_impact = visualization_generators['feature_impact']()
            except Exception as e:
                print(f"Error generating feature impact: {e}")

        if feature_impact:
            elements.append(fig_to_image(feature_impact))
            elements.append(Paragraph("This chart shows how different features contribute to your property's rent value. Green bars indicate positive contributions, red bars indicate negative impacts.", self.styles['ReportNote']))
        else:
            elements.append(Paragraph("Feature impact chart not available", self.styles['ReportNote']))

        elements.append(Spacer(1, 0.2*inch))

        # Add market distribution chart
        elements.append(Paragraph("Market Rent Distribution", self.styles['ReportHeading3']))
        rent_distribution = None

        if 'rent_distribution' in visualizations:
            rent_distribution = visualizations['rent_distribution']
        elif 'rent_distribution' in visualization_generators:
            try:
                rent_distribution = visualization_generators['rent_distribution']()
            except Exception as e:
                print(f"Error generating rent distribution: {e}")

        if rent_distribution:
            elements.append(fig_to_image(rent_distribution))

            # Add percentile info
            percentile = report_data.get('market_position', {}).get('percentile_ranks', {}).get('overall_market', {}).get('rent', 0)
            elements.append(Paragraph(f"This chart shows your property's position in the overall market rent distribution. Your property is at the {percentile:.0f}th percentile of the market.", self.styles['ReportNote']))
        else:
            elements.append(Paragraph("Rent distribution chart not available", self.styles['ReportNote']))

    def decode_category(code, label_encoder):
        # """
        # Decode a category value using a label encoder

        # Parameters:
        # code: The encoded value (int/float)
        # label_encoder: The scikit-learn LabelEncoder that was used for encoding

        # Returns:
        # str: The decoded original string value
        # """
        if not hasattr(label_encoder, 'classes_'):
            return str(code)  # No classes to decode with

        try:
            # Convert to int if it's a float but represents an integer
            if isinstance(code, float) and code.is_integer():
                code = int(code)

            # Check if the code is a valid index
            if isinstance(code, int) and 0 <= code < len(label_encoder.classes_):
                return label_encoder.classes_[code]
        except:
            pass

        return str(code)  # Fall back to string representation
    def _add_summary(self, elements, report_data):
        """Add the summary and recommendations section"""
        # Section header
        # elements.append(PageBreak())
        elements.append(Paragraph("Summary & Recommendations", self.styles['ReportSubtitle']))

        # Get data for summary
        market_position = report_data.get('market_position', {})
        position_category = market_position.get('position_category', 'Unknown')
        premium_discount = market_position.get('premium_discount', {})
        property_details = report_data.get('property_details', {})
        pricing = property_details.get('pricing', {})
        rent = pricing.get('total_rent', 0)

        # Summary text
        summary_text = f"Your property is currently positioned in the <b>{position_category}</b> segment of the market with a monthly rent of Rs. {rent:,}."
        elements.append(Paragraph(summary_text, self.styles['ReportBody']))

        # Get premium/discount from most relevant comparable group
        primary_premium = premium_discount.get('comparables_avg', 0)

        # Add recommendation based on market position
        elements.append(Paragraph("Market Position Insights:", self.styles['ReportHeading3']))

        if position_category in ["Premium", "Above Market"]:
            if primary_premium > 15:
                position_insight = "Your property commands a significant premium over comparable properties. This premium positioning should be supported by maintaining excellent property condition and amenities to justify the higher rent."
            else:
                position_insight = "Your property is positioned above the market average, indicating strong features or amenities that tenants value. This is a good position that balances revenue optimization with occupancy."
        elif position_category == "At Market":
            position_insight = "Your property is priced in line with the market, which should help balance rental income with minimal vacancy periods. This is typically an optimal position for steady income."
        elif position_category in ["Below Market", "Significantly Below Market"]:
            if primary_premium < -15:
                position_insight = "Your property is priced significantly below comparable properties. There may be opportunity to increase rent gradually, especially if you make improvements to the property or when renewing leases."
            else:
                position_insight = "Your property is priced somewhat below the market, which can help minimize vacancy but may mean you're leaving potential rental income on the table."
        else:
            position_insight = "Insufficient data to provide specific market position insights."

        elements.append(Paragraph(position_insight, self.styles['ReportBody']))

        # Add recommendation bullets
        elements.append(Paragraph("Recommendations:", self.styles['ReportHeading3']))

        recommendations = []

        # Generate recommendations based on market position
        if position_category in ["Premium", "Above Market"]:
            recommendations.append("Maintain high standards of property maintenance and amenities to justify the premium rent")
            recommendations.append("Consider investing in property upgrades that will help maintain your competitive advantage")
            if primary_premium > 20:
                recommendations.append("Monitor vacancy periods closely as very high premiums can lead to longer vacancies")
        elif position_category == "At Market":
            recommendations.append("Consider modest rent increases in line with market growth to maintain position")
            recommendations.append("Focus on tenant retention as your pricing is competitive")
            recommendations.append("Monitor market trends to ensure your property stays aligned with comparable properties")
        else:  # Below Market
            recommendations.append("Consider a moderate rent increase with your next lease renewal")
            recommendations.append("Evaluate if there are property improvements that could justify higher rent")
            recommendations.append("Review your tenant acquisition strategy as below-market rent should allow for selectivity")

        # Create bulleted list of recommendations
        rec_items = []
        for rec in recommendations:
            rec_items.append(ListItem(Paragraph(rec, self.styles['ReportBody'])))

        elements.append(ListFlowable(rec_items, bulletType='bullet', leftIndent=20))

        # Add footer note
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph("This analysis is based on current market data and is intended for informational purposes only. Actual market conditions may vary.", self.styles['ReportNote']))


# Example integration function - Call this from your main code
def create_landlord_pdf_report(report_data, label_encoders = None, output_dir="reports", verbose=False):
    # """
    # Create a PDF report from landlord report data

    # Parameters:
    # report_data: Dict - The data structure returned by generate_landlord_report()
    # output_dir: str - Directory to save the PDF report

    # Returns:
    # str: Path to generated PDF file
    # """
    
    
    # Store original print function
    import builtins
    original_print = builtins.print
    
    if not verbose:
        # Suppress print statements during PDF generation
        builtins.print = lambda *args, **kwargs: None
    
    try:
        report_generator = LandlordReportGenerator(output_dir=output_dir, label_encoders=label_encoders)
        pdf_path = report_generator.generate_report(report_data)
    finally:
        # Restore original print function
        builtins.print = original_print

    
    return pdf_path


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import percentileofscore
from datetime import datetime
import shap

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (12, 8)})

def generate_report_for_streamlit(property_data, full_dataset, ml_model=None, 
                                feature_names=None, label_encoders=None, 
                                rent_estimates=None):
    # """
    # Streamlit-friendly wrapper for generate_landlord_report
    # """
    try:
        return generate_landlord_report(
            property_data=property_data,
            full_dataset=full_dataset,
            ml_model=ml_model,
            feature_names=feature_names,
            label_encoders=label_encoders,
            generate_plots=False,  # Generate plots lazily for better performance
            rent_estimates=rent_estimates
        )
    except Exception as e:
        print(f"Error in generate_report_for_streamlit: {e}")
        raise e

# Sample data preprocessing function
def prepare_data(df):
    # """Prepare the dataset for analysis"""
    df = df.copy()

    # Calculate rent per square foot if not already present
    if 'rent_per_sqft' not in df.columns:
        df['rent_per_sqft'] = df['total_rent'] / df['builtup_area'].replace(0, np.nan)

    # Create floor ratio if not already present
    if 'floor_to_total_floors' not in df.columns:
        df['floor_to_total_floors'] = df['floor'] / df['total_floors'].replace(0, 1)

    return df

def find_comparable_properties(property_data, full_dataset, num_comparables=10):
    """
    Find comparable properties prioritizing location hierarchy and then property features

    Parameters:
    property_data: Dict or Series with property features
    full_dataset: DataFrame with all properties
    num_comparables: Minimum number of comparables to return

    Returns:
    Dict: Different categories of comparable properties
    """
    df = full_dataset.copy()

    # Extract key property features
    society = property_data.get('society')
    locality = property_data.get('locality')
    bhk = property_data.get('bedrooms')
    bathrooms = property_data.get('bathrooms')
    area = property_data.get('builtup_area')
    furnishing = property_data.get('furnishing')

    # Additional detailed debugging for society column
    society_values = df['society'].value_counts().head(10)

    # Initialize society_matches to prevent UnboundLocalError
    society_matches = pd.DataFrame()

    # Check for presence of the specific society we're looking for
    if isinstance(society, (int, float)):
        society_matches = df[df['society'] == society]
        # print(f"Debug: Found {len(society_matches)} exact matches for society={society}")

        # If no exact matches, try string conversion
        if len(society_matches) == 0:
            try:
                society_str = str(society)
                society_matches = df[df['society'] == society_str]
                # print(f"Debug: Found {len(society_matches)} matches for society as string='{society_str}'")
                # Use string version if it works better
                if len(society_matches) > 0:
                    society = society_str
                    # print(f"Debug: Switching to string representation of society: '{society}'")
            except:
                pass
    elif isinstance(society, str):
        society_matches = df[df['society'] == society]
        # print(f"Debug: Found {len(society_matches)} exact matches for society='{society}'")

        # If no exact matches, try numeric conversion
        if len(society_matches) == 0:
            try:
                society_num = float(society) if '.' in society else int(society)
                society_matches = df[df['society'] == society_num]
                # print(f"Debug: Found {len(society_matches)} matches for society as number={society_num}")
                # Use numeric version if it works better
                if len(society_matches) > 0:
                    society = society_num
                    # print(f"Debug: Switching to numeric representation of society: {society}")
            except:
                pass

    # If still no matches, find the most similar society value
    if len(society_matches) == 0:
        # print(f"Debug: No exact matches for society. Looking for similar values...")

        # Try case-insensitive match if string
        if isinstance(society, str):
            # Case-insensitive match
            if df['society'].dtype == 'object':  # Only try this with string columns
                society_lower = society.lower()
                for s in df['society'].dropna().unique():
                    if isinstance(s, str) and s.lower() == society_lower:
                        society = s  # Use the version from the dataset
                        # print(f"Debug: Found case-insensitive match: '{society}'")
                        break

    # Define area tolerance (±30%)
    area_min = area * 0.70
    area_max = area * 1.30

    # Location-based hierarchy of comparables with type-safe comparisons

    # 1. Same society, same BHK, similar area
    try:
        same_society_same_bhk = df[(df['society'] == society) &
                                 (df['bedrooms'] == bhk) &
                                 (df['builtup_area'] >= area_min) &
                                 (df['builtup_area'] <= area_max)]
        # print(f"Debug: Found {len(same_society_same_bhk)} properties with same society and BHK")
    except Exception as e:
        # print(f"Error in same_society_same_bhk filter: {e}")
        same_society_same_bhk = pd.DataFrame()

    # 2. Same society, different BHK
    try:
        same_society_diff_bhk = df[(df['society'] == society) &
                                 (df['bedrooms'] != bhk)]
        # print(f"Debug: Found {len(same_society_diff_bhk)} properties with same society, different BHK")
    except Exception as e:
        # print(f"Error in same_society_diff_bhk filter: {e}")
        same_society_diff_bhk = pd.DataFrame()

    # 3. Same locality, same BHK, similar area
    try:
        same_locality_same_bhk = df[(df['locality'] == locality) &
                                  (df['bedrooms'] == bhk) &
                                  (df['builtup_area'] >= area_min) &
                                  (df['builtup_area'] <= area_max)]
        # print(f"Debug: Found {len(same_locality_same_bhk)} properties with same locality and BHK")

        # If we have too many, try to refine to exclude properties from different societies
        if len(same_locality_same_bhk) > 50:
            try:
                refined = same_locality_same_bhk[same_locality_same_bhk['society'] != society]
                if len(refined) > 0:
                    same_locality_same_bhk = refined
                    # print(f"Debug: Refined to {len(same_locality_same_bhk)} properties excluding our society")
            except:
                pass
    except Exception as e:
        # print(f"Error in same_locality_same_bhk filter: {e}")
        same_locality_same_bhk = pd.DataFrame()

    # 4. Same locality, similar BHK (±1)
    try:
        same_locality_similar_bhk = df[(df['locality'] == locality) &
                                     (df['bedrooms'].between(bhk-1, bhk+1))]
        # print(f"Debug: Found {len(same_locality_similar_bhk)} properties with same locality, similar BHK")
    except Exception as e:
        # print(f"Error in same_locality_similar_bhk filter: {e}")
        same_locality_similar_bhk = pd.DataFrame()

    # If all filters failed, try a more permissive approach
    if (len(same_society_same_bhk) == 0 and len(same_society_diff_bhk) == 0 and
        len(same_locality_same_bhk) == 0 and len(same_locality_similar_bhk) == 0):
        # print("Debug: All regular filters returned no results, trying more permissive filters")

        # Try to find at least some comparables by BHK
        try:
            bhk_comparables = df[df['bedrooms'] == bhk]
            # print(f"Debug: Found {len(bhk_comparables)} properties with same BHK")

            if len(bhk_comparables) > 0:
                same_locality_same_bhk = bhk_comparables
        except Exception as e:
            print(f"Error in fallback BHK filter: {e}")

    # SPECIAL CASE: If same_locality_same_bhk is too large (1000+ properties),
    # we should create a more focused subset to use as the primary comparison group
    if len(same_locality_same_bhk) > 1000:
        print(f"Debug: same_locality_same_bhk is very large ({len(same_locality_same_bhk)} properties), creating a more focused subset")

        # Create a more focused subset based on similar area
        focused_comparable_area = same_locality_same_bhk[
            (same_locality_same_bhk['builtup_area'] >= area * 0.85) &
            (same_locality_same_bhk['builtup_area'] <= area * 1.15)
        ]

        # If we have a good number of properties in the focused subset, use it
        if len(focused_comparable_area) >= 20:
            # print(f"Debug: Created a focused subset with {len(focused_comparable_area)} properties based on similar area")
            # Create a new comparison group for this focused subset
            comparables_dict = {
                'focused_locality_area': focused_comparable_area,
                'same_society_same_bhk': same_society_same_bhk,
                'same_locality_same_bhk': same_locality_same_bhk,
                'same_society_diff_bhk': same_society_diff_bhk,
                'same_locality_similar_bhk': same_locality_similar_bhk
            }

            # Calculate similarity score for fine-grained ranking within each group
            def calculate_similarity(group_df):
                if len(group_df) == 0:
                    return group_df

                result = group_df.copy()
                result['similarity_score'] = 0

                # Area similarity (higher weight)
                if 'builtup_area' in result.columns:
                    max_area_diff = area * 0.30  # 30% difference max
                    result['area_diff'] = abs(result['builtup_area'] - area)
                    result['area_score'] = 1 - (result['area_diff'] / max_area_diff).clip(upper=1)
                    result['similarity_score'] += 0.30 * result['area_score']  # 30% weight to area

                # BHK exact match
                if 'bedrooms' in result.columns:
                    result['bhk_score'] = (result['bedrooms'] == bhk).astype(float)
                    result['similarity_score'] += 0.40 * result['bhk_score']  # 40% weight to BHK

                # Bathroom similarity
                if 'bathrooms' in result.columns and bathrooms is not None:
                    result['bath_score'] = (result['bathrooms'] == bathrooms).astype(float)
                    result['similarity_score'] += 0.15 * result['bath_score']  # 15% weight to bathrooms

                # Furnishing match - with type safety
                if 'furnishing' in result.columns and furnishing is not None:
                    try:
                        result['furnishing_score'] = (result['furnishing'] == furnishing).astype(float)
                        result['similarity_score'] += 0.2 * result['furnishing_score']  # 15% weight to furnishing
                    except Exception as e:
                        print(f"Error comparing furnishing: {e}")
                        # Fallback - no furnishing score
                        result['furnishing_score'] = 0

                # Sort by similarity score
                result = result.sort_values('similarity_score', ascending=False)

                # Drop intermediate score columns
                cols_to_drop = ['area_diff', 'area_score', 'bhk_score', 'bath_score', 'furnishing_score']
                result = result.drop([col for col in cols_to_drop if col in result.columns], axis=1)

                return result

            # Apply similarity scoring to each group
            for key in comparables_dict:
                comparables_dict[key] = calculate_similarity(comparables_dict[key])

            # Exclude the property itself if it's in the dataset
            if 'property_id' in property_data and 'property_id' in df.columns:
                property_id = property_data.get('property_id')
                for key in comparables_dict:
                    if len(comparables_dict[key]) > 0:
                        comparables_dict[key] = comparables_dict[key][comparables_dict[key]['property_id'] != property_id]

            # Combine all comparables in order of relevance
            all_comparables = pd.concat([
                comparables_dict['focused_locality_area'].assign(comparable_type='focused_locality_area'),
                comparables_dict['same_society_same_bhk'].assign(comparable_type='same_society_same_bhk'),
                comparables_dict['same_locality_same_bhk'].assign(comparable_type='same_locality_same_bhk'),
                comparables_dict['same_society_diff_bhk'].assign(comparable_type='same_society_diff_bhk'),
                comparables_dict['same_locality_similar_bhk'].assign(comparable_type='same_locality_similar_bhk')
            ]).drop_duplicates()

            # Take the top comparables
            top_comparables = all_comparables.head(num_comparables)

            return {
                'top_comparables': top_comparables,
                'focused_locality_area': comparables_dict['focused_locality_area'],
                'same_society_same_bhk': comparables_dict['same_society_same_bhk'],
                'same_locality_same_bhk': comparables_dict['same_locality_same_bhk'],
                'same_society_diff_bhk': comparables_dict['same_society_diff_bhk'],
                'same_locality_similar_bhk': comparables_dict['same_locality_similar_bhk'],
                'all_comparables': all_comparables
            }

    # Calculate similarity score for fine-grained ranking within each group
    def calculate_similarity(group_df):
        if len(group_df) == 0:
            return group_df

        result = group_df.copy()
        result['similarity_score'] = 0

        # Area similarity (higher weight)
        if 'builtup_area' in result.columns:
            max_area_diff = area * 0.30  # 30% difference max
            result['area_diff'] = abs(result['builtup_area'] - area)
            result['area_score'] = 1 - (result['area_diff'] / max_area_diff).clip(upper=1)
            result['similarity_score'] += 0.30 * result['area_score']  # 30% weight to area

        # BHK exact match
        if 'bedrooms' in result.columns:
            result['bhk_score'] = (result['bedrooms'] == bhk).astype(float)
            result['similarity_score'] += 0.40 * result['bhk_score']  # 40% weight to BHK

        # Bathroom similarity
        if 'bathrooms' in result.columns and bathrooms is not None:
            result['bath_score'] = (result['bathrooms'] == bathrooms).astype(float)
            result['similarity_score'] += 0.15 * result['bath_score']  # 15% weight to bathrooms

        # Furnishing match - with type safety
        if 'furnishing' in result.columns and furnishing is not None:
            try:
                result['furnishing_score'] = (result['furnishing'] == furnishing).astype(float)
                result['similarity_score'] += 0.2 * result['furnishing_score']  # 15% weight to furnishing
            except Exception as e:
                print(f"Error comparing furnishing: {e}")
                # Fallback - no furnishing score
                result['furnishing_score'] = 0

        # Sort by similarity score
        result = result.sort_values('similarity_score', ascending=False)

        # Drop intermediate score columns
        cols_to_drop = ['area_diff', 'area_score', 'bhk_score', 'bath_score', 'furnishing_score']
        result = result.drop([col for col in cols_to_drop if col in result.columns], axis=1)

        return result

    # Apply similarity scoring to each group
    same_society_same_bhk = calculate_similarity(same_society_same_bhk)
    same_society_diff_bhk = calculate_similarity(same_society_diff_bhk)
    same_locality_same_bhk = calculate_similarity(same_locality_same_bhk)
    same_locality_similar_bhk = calculate_similarity(same_locality_similar_bhk)

    # Exclude the property itself if it's in the dataset
    if 'property_id' in property_data and 'property_id' in df.columns:
        property_id = property_data.get('property_id')
        same_society_same_bhk = same_society_same_bhk[same_society_same_bhk['property_id'] != property_id]
        same_society_diff_bhk = same_society_diff_bhk[same_society_diff_bhk['property_id'] != property_id]
        same_locality_same_bhk = same_locality_same_bhk[same_locality_same_bhk['property_id'] != property_id]
        same_locality_similar_bhk = same_locality_similar_bhk[same_locality_similar_bhk['property_id'] != property_id]

    # Create fallback top_comparables in case no good comparables found
    if (len(same_society_same_bhk) == 0 and len(same_locality_same_bhk) == 0 and
        len(same_society_diff_bhk) == 0 and len(same_locality_similar_bhk) == 0):
        print("Debug: No good comparables found, creating a fallback set")

        # Create a fallback based on BHK
        try:
            fallback_comparables = df[df['bedrooms'] == bhk].head(num_comparables)
            if len(fallback_comparables) < num_comparables:
                # If not enough, just take any properties
                fallback_comparables = df.head(num_comparables)
        except:
            fallback_comparables = df.head(num_comparables)

        # print(f"Debug: Created fallback set with {len(fallback_comparables)} properties")
        top_comparables = fallback_comparables.assign(comparable_type='fallback')
        all_comparables = top_comparables
    else:
        # Combine all comparables in order of relevance
        all_comparables = pd.concat([
            same_society_same_bhk.assign(comparable_type='same_society_same_bhk'),
            same_locality_same_bhk.assign(comparable_type='same_locality_same_bhk'),
            same_society_diff_bhk.assign(comparable_type='same_society_diff_bhk'),
            same_locality_similar_bhk.assign(comparable_type='same_locality_similar_bhk')
        ]).drop_duplicates()

        # Take the top comparables
        top_comparables = all_comparables.head(num_comparables)

        # print(f"Debug: Created top_comparables with {len(top_comparables)} properties")

    return {
        'top_comparables': top_comparables,
        'same_society_same_bhk': same_society_same_bhk,
        'same_locality_same_bhk': same_locality_same_bhk,
        'same_society_diff_bhk': same_society_diff_bhk,
        'same_locality_similar_bhk': same_locality_similar_bhk,
        'all_comparables': all_comparables
    }

def calculate_market_position(property_data, comparables_dict, full_dataset):
    # """
    # Calculate the market position of the property using hierarchical comparables - with improved error handling

    # Parameters:
    # property_data: Series or dict with target property features
    # comparables_dict: Dictionary with different tiers of comparable properties
    # full_dataset: Full dataset for broader market comparison

    # Returns:
    # dict: Market position metrics
    # """
    # Convert property data to standardized format if needed
    if not isinstance(property_data, pd.Series):
        property_data = pd.Series(property_data)

    property_rent = property_data.get('total_rent')
    property_rent_sqft = property_data.get('rent_per_sqft')

    # Extract the most relevant comparables in priority order
    # Include the new focused_locality_area group if available
    comp_groups = ['focused_locality_area', 'same_society_same_bhk', 'same_locality_same_bhk',
                   'same_society_diff_bhk', 'same_locality_similar_bhk']


    # Find the first non-empty group to use as primary comparables
    primary_comparables = None
    primary_group = None
    for group in comp_groups:
        if group in comparables_dict and len(comparables_dict[group]) > 0:
            primary_comparables = comparables_dict[group]
            primary_group = group
            print(f"Selected primary group: {primary_group} with {len(primary_comparables)} properties")
            break

    # If no good comparables found, use all_comparables
    if primary_comparables is None or len(primary_comparables) == 0:
        primary_comparables = comparables_dict.get('all_comparables', pd.DataFrame())
        primary_group = 'all_comparables'
        print(f"Fallback to all_comparables group with {len(primary_comparables)} properties")

    # Fallback to top_comparables if all_comparables is empty
    if len(primary_comparables) == 0 and 'top_comparables' in comparables_dict:
        primary_comparables = comparables_dict['top_comparables']
        primary_group = 'top_comparables'
        print(f"Fallback to top_comparables group with {len(primary_comparables)} properties")

    # CRITICAL FIX: If we still don't have a primary group, set to a default value
    if primary_group is None:
        primary_group = 'overall_market'
        print("Warning: No valid comparison group found, using 'overall_market' as fallback")

    # Calculate percentile ranks for different segments
    def safe_percentile(series, value):
        if len(series) == 0 or pd.isna(value):
            return np.nan
        return percentileofscore(series.dropna(), value)

    # Overall market percentiles
    locality = property_data.get('locality')
    bedrooms = property_data.get('bedrooms')

    locality_properties = full_dataset[full_dataset['locality'] == locality]
    bedrooms_filter = full_dataset['bedrooms'] == bedrooms

    # Calculate percentile ranks for different comparison groups
    percentile_ranks = {
        'overall_market': {
            'rent': safe_percentile(full_dataset['total_rent'], property_rent),
            'rent_sqft': safe_percentile(full_dataset['rent_per_sqft'], property_rent_sqft)
        },
        'locality': {
            'rent': safe_percentile(locality_properties['total_rent'], property_rent),
            'rent_sqft': safe_percentile(locality_properties['rent_per_sqft'], property_rent_sqft)
        },
        'bedroom_type': {
            'rent': safe_percentile(full_dataset[bedrooms_filter]['total_rent'], property_rent),
            'rent_sqft': safe_percentile(full_dataset[bedrooms_filter]['rent_per_sqft'], property_rent_sqft)
        }
    }

    # Add percentiles for each comparable group
    for group in comp_groups:
        if group in comparables_dict and len(comparables_dict[group]) > 0:
            group_df = comparables_dict[group]
            percentile_ranks[group] = {
                'rent': safe_percentile(group_df['total_rent'], property_rent),
                'rent_sqft': safe_percentile(group_df['rent_per_sqft'], property_rent_sqft)
            }

    # Calculate premium/discount vs different segments
    def calculate_premium(value, reference):
        if reference == 0 or pd.isna(reference) or pd.isna(value):
            return 0
        return ((value - reference) / reference) * 100

    premium_discount = {
        'locality_avg': calculate_premium(property_rent_sqft, locality_properties['rent_per_sqft'].mean()),
        'bedroom_type_avg': calculate_premium(property_rent_sqft, full_dataset[bedrooms_filter]['rent_per_sqft'].mean())
    }

    # Add premium/discount for each comparable group
    for group in comp_groups:
        if group in comparables_dict and len(comparables_dict[group]) > 0:
            group_df = comparables_dict[group]
            premium_discount[f"{group}_avg"] = calculate_premium(property_rent_sqft, group_df['rent_per_sqft'].mean())

    # Always add a 'comparables_avg' key that maps to the primary comparison group
    if primary_group and f"{primary_group}_avg" in premium_discount:
        premium_discount['comparables_avg'] = premium_discount[f"{primary_group}_avg"]
    else:
        premium_discount['comparables_avg'] = premium_discount.get('locality_avg', 0)
        print("Warning: Using locality_avg as fallback for comparables_avg")

    # Determine market position category based on primary comparables
    if primary_group is not None and primary_group in percentile_ranks:
        comp_percentile = percentile_ranks[primary_group].get('rent_sqft', 50)
    else:
        comp_percentile = percentile_ranks['overall_market'].get('rent_sqft', 50)
        print("Warning: Using overall_market percentile as fallback")

    if pd.isna(comp_percentile):
        position_category = "Insufficient Data"
    elif comp_percentile >= 90:
        position_category = "Premium"
    elif comp_percentile >= 60:
        position_category = "Above Market"
    elif comp_percentile >= 40:
        position_category = "At Market"
    elif comp_percentile >= 10:
        position_category = "Below Market"
    else:
        position_category = "Significantly Below Market"

    # Print summary for debugging
    print(f"Market position: {position_category}")
    print(f"Percentile: {comp_percentile:.1f}")
    print(f"Primary group premium: {premium_discount.get(f'{primary_group}_avg', 0):.1f}%")
    print(f"Comparables avg premium: {premium_discount.get('comparables_avg', 0):.1f}%")

    # Add information about which comparison group was used for positioning
    return {
        'percentile_ranks': percentile_ranks,
        'premium_discount': premium_discount,
        'position_category': position_category,
        'primary_comparison_group': primary_group,
        'num_primary_comparables': len(primary_comparables) if primary_comparables is not None else 0
    }

def create_property_position_chart(property_data, comparables_dict):
    # """
    # Create a chart showing property position among comparables

    # Parameters:
    # property_data: Series with property features
    # comparables_dict: Dictionary with different tiers of comparable properties

    # Returns:
    # Figure: Matplotlib figure object
    # """
    property_rent = property_data.get('total_rent')

    # Determine which comparables to use, in priority order
    comp_groups = ['same_society_same_bhk', 'same_locality_same_bhk',
                  'same_society_diff_bhk', 'same_locality_similar_bhk']

    # Find the first non-empty group to visualize
    primary_comparables = None
    group_label = None
    for group in comp_groups:
        if group in comparables_dict and len(comparables_dict[group]) > 0:
            primary_comparables = comparables_dict[group]
            group_label = group.replace('_', ' ').title()
            break

    # Fallback to all comparables if no specific group is available
    if primary_comparables is None or len(primary_comparables) == 0:
        primary_comparables = comparables_dict.get('all_comparables',
                                                 comparables_dict.get('top_comparables', pd.DataFrame()))
        group_label = "All Comparables"

    # Create the visualization
    plt.figure(figsize=(10, 6))

    if len(primary_comparables) > 0:
        # Plot histogram of comparable rents
        sns.histplot(primary_comparables['total_rent'], bins=min(10, len(primary_comparables)),
                    kde=True, alpha=0.6)

        # Add vertical line for property rent
        plt.axvline(x=property_rent, color='red', linestyle='--', linewidth=2,
                   label=f'Your Property (Rs. {property_rent:,.0f})')

        # Add vertical line for average rent
        avg_rent = primary_comparables['total_rent'].mean()
        plt.axvline(x=avg_rent, color='green', linestyle='-', linewidth=2,
                   label=f'Average (Rs. {avg_rent:,.0f})')

        plt.title(f'Your Property vs. {group_label}', fontsize=14)
        plt.xlabel('Monthly Rent (Rs.)', fontsize=12)
        plt.ylabel('Number of Properties', fontsize=12)
        plt.legend()
    else:
        # Handle case with no comparables
        plt.text(0.5, 0.5, "Insufficient comparable properties found",
                ha='center', va='center', fontsize=14)
        plt.title('Property Comparison', fontsize=14)

    plt.tight_layout()

    return plt.gcf()

def create_feature_comparison_radar(property_data, comparables_dict, full_dataset):
    # """
    # Create an enhanced radar chart comparing property features to various comparable groups
    # with additional market positioning metrics

    # Parameters:
    # property_data: Dict or Series with property features
    # comparables_dict: Dictionary with different tiers of comparable properties
    # full_dataset: Complete dataset for percentile calculations
    # """
    # Select basic features for radar chart
    basic_features = ['builtup_area', 'floor_to_total_floors', 'bathrooms', 'rent_per_sqft']

    # Add market positioning metrics
    market_metrics = ['market_percentile', 'price_premium']
    features = basic_features + market_metrics

    # Get different comparison groups
    same_society_same_bhk = comparables_dict.get('same_society_same_bhk', pd.DataFrame())
    same_locality_same_bhk = comparables_dict.get('same_locality_same_bhk', pd.DataFrame())
    all_comparables = comparables_dict.get('all_comparables', pd.DataFrame())

    # Initialize data structures for all comparison groups
    comp_groups = {
        'all_comparables': all_comparables
    }

    # Only include groups with sufficient data
    if len(same_society_same_bhk) >= 3:
        comp_groups['same_society_same_bhk'] = same_society_same_bhk

    if len(same_locality_same_bhk) >= 3:
        comp_groups['same_locality_same_bhk'] = same_locality_same_bhk

    # Calculate averages for all comparison groups
    comp_avg = {group_name: {} for group_name in comp_groups.keys()}

    # Calculate basic feature averages
    for group_name, group_df in comp_groups.items():
        for feature in basic_features:
            if feature in group_df.columns:
                comp_avg[group_name][feature] = group_df[feature].mean()
            else:
                comp_avg[group_name][feature] = 0

    # Calculate market percentiles for the property
    bhk = property_data.get('bedrooms')
    comparable_bhk_props = full_dataset[full_dataset['bedrooms'] == bhk]

    # Calculate property's market percentile for each basic feature
    property_percentiles = {}
    for feature in basic_features:
        if feature in comparable_bhk_props.columns:
            property_value = property_data.get(feature)
            property_percentiles[feature] = percentileofscore(comparable_bhk_props[feature].dropna(), property_value)
        else:
            property_percentiles[feature] = 50  # Default to median if data unavailable

    # Add market percentile to property data and all comparison groups (at 50th percentile by definition)
    property_data['market_percentile'] = np.mean(list(property_percentiles.values()))
    for group_name in comp_avg.keys():
        comp_avg[group_name]['market_percentile'] = 50

    # Calculate price premium/discount
    property_rent = property_data.get('total_rent')
    property_data['price_premium'] = 0  # Will be calculated below

    for group_name, group_df in comp_groups.items():
        if 'total_rent' in group_df.columns and len(group_df) > 0:
            avg_rent = group_df['total_rent'].mean()
            if avg_rent > 0:
                # Calculate premium as percentage difference from comparable average
                premium_pct = ((property_rent - avg_rent) / avg_rent) * 100

                # Normalize to a 0-100 scale for radar chart
                # 0 means 50% below market (big discount)
                # 50 means at market
                # 100 means 50% above market (big premium)
                normalized_premium = 50 + premium_pct
                normalized_premium = max(0, min(100, normalized_premium))

                if group_name == 'all_comparables':
                    property_data['price_premium'] = normalized_premium

                comp_avg[group_name]['price_premium'] = 50  # Neutral by definition
        else:
            comp_avg[group_name]['price_premium'] = 50

    # Normalize values between 0-1 for comparison
    max_values = {}
    for feature in features:
        feature_values = [property_data.get(feature)]
        for group_vals in comp_avg.values():
            if feature in group_vals:
                feature_values.append(group_vals[feature])

        max_val = max(feature_values) * 1.2  # 20% buffer
        max_values[feature] = max_val if max_val > 0 else 1  # Avoid division by zero

    # Normalize property values
    property_norm = {}
    for feature in features:
        property_value = property_data.get(feature, 0)
        property_norm[feature] = property_value / max_values[feature]

    # Normalize comparison group values
    comp_avg_norm = {group_name: {} for group_name in comp_groups.keys()}
    for group_name in comp_groups.keys():
        for feature in features:
            if feature in comp_avg[group_name]:
                comp_avg_norm[group_name][feature] = comp_avg[group_name][feature] / max_values[feature]
            else:
                comp_avg_norm[group_name][feature] = 0

    # Create radar chart
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)

    # Set the angles for each feature
    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Add property values
    property_values = [property_norm[feature] for feature in features]
    property_values += property_values[:1]  # Close the loop

    # Plot the property
    ax.plot(angles, property_values, 'r-', linewidth=2.5, label='Your Property')
    ax.fill(angles, property_values, 'r', alpha=0.1)

    # Define colors and styles for different comparison groups
    styles = {
        'same_society_same_bhk': {'color': 'g', 'linestyle': '-', 'label': 'Same Society & BHK'},
        'same_locality_same_bhk': {'color': 'b', 'linestyle': '-', 'label': 'Same Locality & BHK'},
        'all_comparables': {'color': 'purple', 'linestyle': '--', 'label': 'All Comparables'}
    }

    # Plot each comparison group
    for group_name, style in styles.items():
        if group_name in comp_avg_norm:
            group_values = [comp_avg_norm[group_name].get(feature, 0) for feature in features]
            group_values += group_values[:1]  # Close the loop

            ax.plot(angles, group_values,
                    color=style['color'],
                    linestyle=style['linestyle'],
                    linewidth=2,
                    label=style['label'])
            ax.fill(angles, group_values, style['color'], alpha=0.05)

    # Add feature labels with descriptions in parentheses
    feature_labels = []
    for feature in features:
        if feature == 'market_percentile':
            label = f"Market Percentile ({property_percentiles.get('rent_per_sqft', 0):.0f}th)"
        elif feature == 'price_premium':
            premium = property_data.get('price_premium', 50) - 50
            if abs(premium) < 5:
                label = "Price (At Market)"
            elif premium > 0:
                label = f"Price (+{premium:.0f}% Premium)"
            else:
                label = f"Price ({premium:.0f}% Discount)"
        else:
            label = feature.replace('_', ' ').title()
        feature_labels.append(label)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_labels, fontsize=9)

    # Add radial grid labels (0.2, 0.4, 0.6, 0.8, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.set_rlabel_position(90)  # Move radial labels to 90 degrees

    plt.title('Property Comparison: Features and Market Position', fontsize=14)
    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout()

    return fig

def plot_rent_distribution_with_property(property_data, full_dataset):
    # """Plot the rent distribution with the property's position highlighted"""
    plt.figure(figsize=(10, 6))

    # Plot overall rent distribution
    sns.histplot(full_dataset['total_rent'], bins=30, kde=True, alpha=0.6)

    # Add vertical line for property rent
    property_rent = property_data.get('total_rent')
    plt.axvline(x=property_rent, color='red', linestyle='--', linewidth=2,
                label=f'Your Property (Rs. {property_rent:,.0f})')

    # Add vertical line for market average
    market_avg = full_dataset['total_rent'].mean()
    plt.axvline(x=market_avg, color='green', linestyle='-', linewidth=2,
                label=f'Market Average (Rs. {market_avg:,.0f})')

    plt.title('Your Property in the Overall Rental Market', fontsize=14)
    plt.xlabel('Monthly Rent (Rs.)', fontsize=12)
    plt.ylabel('Number of Properties', fontsize=12)
    plt.legend()
    plt.tight_layout()

    return plt.gcf()

def create_robust_shap_waterfall_chart(property_data, ml_model, feature_names, label_encoders=None, target_prediction=None):
    # """
    # Create a robust waterfall chart with proper encoding of categorical variables
    # and scale it to match a target prediction value. Display from top to bottom.
    # """
    try:
        # Basic validation
        if ml_model is None or feature_names is None:
            print("Model or feature names is None, using dummy chart")
            return dummy_waterfall_chart(property_data, target_prediction)

        # Ensure we're only using features the model knows about
        model_features = list(ml_model.feature_names_in_) if hasattr(ml_model, 'feature_names_in_') else feature_names

        # Use the correct feature list
        feature_names = model_features

        # Create a copy of property data
        property_data_copy = property_data.copy()
        input_df = pd.DataFrame([property_data_copy])

        # Create model input with exact columns model expects
        model_input = pd.DataFrame(index=[0], columns=feature_names)

        # Process categorical features with proper encoding
        categorical_features = ['furnishing', 'locality', 'society']
        for feature in categorical_features:
            if feature in feature_names:
                if feature in input_df.columns and label_encoders and feature in label_encoders:
                    try:
                        feature_value = input_df[feature].iloc[0]

                        # Check if already numeric (pre-encoded)
                        if isinstance(feature_value, (int, float, np.number)):
                            model_input[feature] = feature_value
                        else:
                            # Need to encode string value
                            feature_str = str(feature_value)

                            # Check if value exists in encoder classes
                            if feature_str in label_encoders[feature].classes_:
                                encoded_value = label_encoders[feature].transform([feature_str])[0]
                                model_input[feature] = encoded_value
                            else:
                                # Try case-insensitive match
                                feature_lower = feature_str.lower()
                                for i, cls in enumerate(label_encoders[feature].classes_):
                                    if cls.lower() == feature_lower:
                                        model_input[feature] = i
                                        break
                                else:
                                    # No match found, use default (0)
                                    model_input[feature] = 0
                    except Exception as e:
                        print(f"Error encoding {feature}: {e}")
                        model_input[feature] = 0
                else:
                    model_input[feature] = 0
            
        # Process numeric features
        for feature in feature_names:
            if feature not in categorical_features:
                if feature == 'log_builtup_area':
                    # Special handling for log_builtup_area
                    if 'builtup_area' in input_df.columns:
                        area_val = input_df['builtup_area'].iloc[0]
                        model_input[feature] = np.log1p(area_val)
                    elif 'log_builtup_area' in input_df.columns:
                        model_input[feature] = input_df['log_builtup_area'].iloc[0]
                    else:
                        model_input[feature] = 0
                elif feature in input_df.columns:
                    model_input[feature] = input_df[feature].iloc[0]
                else:
                    # Try derived features
                    if feature == 'floor_to_total_floors' and 'floor' in input_df.columns and 'total_floors' in input_df.columns:
                        floor = input_df['floor'].iloc[0]
                        total_floors = input_df['total_floors'].iloc[0]
                        if total_floors > 0:
                            model_input[feature] = floor / total_floors
                        else:
                            model_input[feature] = 0
                    elif feature == 'building_age' and 'building_age' not in input_df.columns:
                        # Default to 5 years if not provided
                        model_input[feature] = 5
                    else:
                        model_input[feature] = 0

        # Ensure all values are numeric
        for col in model_input.columns:
            if not pd.api.types.is_numeric_dtype(model_input[col]):
                try:
                    model_input[col] = pd.to_numeric(model_input[col])
                except:
                    model_input[col] = 0

        # Get the model's direct prediction for scaling factor calculation
        try:
            direct_pred = ml_model.predict(model_input)[0]
            print(f"Original model prediction: {direct_pred:.2f}")
        except Exception as e:
            print(f"Warning: Direct prediction failed: {e}")
            direct_pred = property_data.get('total_rent', 25000)

        # Create SHAP explainer
        explainer = shap.TreeExplainer(ml_model)

        # Get expected value (base value)
        try:
            base_value = explainer.expected_value
            if hasattr(base_value, "__len__"):
                base_value = base_value[0]
        except Exception as e:
            print(f"Could not get base value: {e}, using fallback")
            base_value = property_data.get('total_rent', 25000) * 0.7  # Fallback estimate

        # Calculate SHAP values
        shap_values = explainer(model_input)

        # Handle different SHAP library versions
        if hasattr(shap_values, "values"):
            shap_vals = shap_values.values[0]
            feature_names_for_display = shap_values.feature_names
        else:
            shap_vals = shap_values[0] if hasattr(shap_values, "__len__") else shap_values
            feature_names_for_display = feature_names

        # Create DataFrames for plotting
        waterfall_data = pd.DataFrame({
            'feature': feature_names_for_display,
            'shap_value': shap_vals
        })

        # Remove features with NaN or zero impact
        waterfall_data = waterfall_data.dropna(subset=['shap_value'])
        waterfall_data = waterfall_data[waterfall_data['shap_value'] != 0]

        if len(waterfall_data) == 0:
            print("No non-zero SHAP values, using dummy chart")
            return dummy_waterfall_chart(property_data, target_prediction)

        # Sort by absolute impact
        waterfall_data['abs_shap'] = waterfall_data['shap_value'].abs()
        waterfall_data = waterfall_data.sort_values('abs_shap', ascending=False)

        # Format display names - Replace log_ and decode categorical values
        display_names = []
        for idx, row in waterfall_data.iterrows():
            feature = row['feature']
            disp_name = feature.replace('log_', '')

            # Add decoded categorical values if available
            if disp_name in categorical_features and label_encoders and disp_name in label_encoders:
                try:
                    feature_value = model_input[feature].iloc[0]
                    if hasattr(label_encoders[disp_name], 'inverse_transform'):
                        original_value = label_encoders[disp_name].inverse_transform([int(feature_value)])[0]
                        disp_name = f"{disp_name}: {original_value}"
                except Exception as e:
                    print(f"Could not decode {disp_name}: {e}")

            # Format name
            disp_name = disp_name.replace('_', ' ').title()
            display_names.append(disp_name)

        waterfall_data['display_name'] = display_names

        # Get feature and impact lists
        features = waterfall_data['display_name'].tolist()
        impacts = waterfall_data['shap_value'].tolist()

        # Calculate total impact and prediction from SHAP values
        total_impact = sum(impacts)
        original_prediction = base_value + total_impact
        
        # Calculate scaling factor if target prediction is provided
        if target_prediction is not None and original_prediction > 0:
            # Calculate scaling factor for SHAP values
            scaling_factor = target_prediction / original_prediction
            print(f"Scaling factor: {scaling_factor:.4f} (Target: {target_prediction:.2f} / Original: {original_prediction:.2f})")
            
            # Scale the impacts and base value
            scaled_impacts = [impact * scaling_factor for impact in impacts]
            scaled_base_value = base_value * scaling_factor
            scaled_prediction = scaled_base_value + sum(scaled_impacts)
            
            print(f"Scaled base value: {scaled_base_value:.2f}")
            print(f"Scaled prediction: {scaled_prediction:.2f} (Target: {target_prediction:.2f})")
        else:
            scaled_impacts = impacts
            scaled_base_value = base_value
            scaled_prediction = original_prediction




        import matplotlib.pyplot as plt
                # ===== IMPROVED WATERFALL CHART =====
        plt.style.use('seaborn-v0_8-whitegrid')  # Use a cleaner style
        
        # Create figure with appropriate height and width
        fig_height = max(10, (len(features) + 2) * 0.6)  # More space per row
        fig = plt.figure(figsize=(14, fig_height))
        ax = fig.add_subplot(111)
        
        # Set background color
        ax.set_facecolor('#f0f0f0')
        
        # All labels (Base → Features → Final)
        all_labels = ['Base Value'] + features + ['Final Prediction']
        
        # Calculate positions for the waterfall segments
        values = [scaled_base_value] + scaled_impacts + [0]  # Add 0 for final prediction
        
        # Calculate the cumulative sum at each step
        cumulative = np.zeros(len(values))
        cumulative[0] = values[0]
        for i in range(1, len(values)-1):
            cumulative[i] = cumulative[i-1] + values[i]
        cumulative[-1] = scaled_prediction  # Set final prediction
        
        # Calculate the starting positions
        bottoms = np.zeros(len(values))
        bottoms[0] = 0  # Base value starts at 0
        for i in range(1, len(values)-1):
            if values[i] >= 0:
                bottoms[i] = cumulative[i-1]  # Positive values start from previous total
            else:
                bottoms[i] = cumulative[i-1] + values[i]  # Negative values end at previous total
        bottoms[-1] = 0  # Final prediction starts at 0
        
        # Create custom colors
        colors = []
        for i, val in enumerate(values):
            if i == 0:  # Base value
                colors.append('#5DA5DA')  # Light blue
            elif i == len(values)-1:  # Final prediction
                colors.append('#000080')  # Navy blue
            elif val >= 0:  # Positive impact
                colors.append('#60B05C')  # Green
            else:  # Negative impact
                colors.append('#D45C43')  # Red
        
        # Determine bar widths - thinner for features with small impact
        widths = []
        for val in values:
            if abs(val) < 0.02 * scaled_prediction:
                widths.append(0.6)
            else:
                widths.append(0.8)
        
        # Plot the bars
        bars = []
        for i, (val, bottom, color, width) in enumerate(zip(values, bottoms, colors, widths)):
            if i == len(values)-1:  # Final prediction
                # For final prediction, use cumulative sum (not the 0 we added)
                bar = ax.barh(i, cumulative[-1], left=0, color=color, height=width, edgecolor='black', linewidth=0.5)
            else:
                bar = ax.barh(i, abs(val), left=bottom, color=color, height=width, edgecolor='black', linewidth=0.5)
            bars.append(bar)
        
        # Add connecting lines between bars
        for i in range(len(values)-1):
            if i == 0:  # From base to first feature
                ax.plot([cumulative[0], cumulative[0]], [i+0.4, i+0.6], 'k-', linewidth=1.5)
            elif values[i+1] == 0:  # Skip if next value is 0 (final prediction)
                continue
            else:
                if values[i] >= 0:
                    # Connect from top of this bar to start of next bar
                    ax.plot([cumulative[i], cumulative[i]], [i+0.4, i+1-0.4], 'k-', linewidth=1.5)
                else:
                    # Connect from bottom of this bar to start of next bar
                    ax.plot([cumulative[i-1], cumulative[i-1]], [i+0.4, i+1-0.4], 'k-', linewidth=1.5)
        
        # Connect last feature to final prediction
        ax.plot([cumulative[-2], cumulative[-2]], [len(values)-2+0.4, len(values)-1-0.4], 'k-', linewidth=1.5)
        
        # Add value labels to each bar
        for i, (val, bottom) in enumerate(zip(values, bottoms)):
            if i == 0:  # Base value
                # Position text in middle of bar
                ax.text(val/2, i, f"Rs. {val:,.0f}", ha='center', va='center', 
                       fontweight='bold', color='white' if val > 30000 else 'black')
            elif i == len(values)-1:  # Final prediction
                # Position text in middle of bar
                ax.text(cumulative[-1]/2, i, f"Rs. {cumulative[-1]:,.0f}", ha='center', va='center', 
                       fontweight='bold', color='white')
            else:
                # Only show significant contributions (to avoid clutter)
                if abs(val) > 0.01 * scaled_prediction:  
                    # Format: +Rs. X,XXX or -Rs. X,XXX
                    text = f"+Rs. {val:,.0f}" if val > 0 else f"Rs. {val:,.0f}"
                    
                    # Position text at right end of bar for space
                    if val > 0:
                        text_x = bottom + val + (0.01 * scaled_prediction)
                        ha = 'left'
                    else:
                        text_x = bottom - (0.01 * scaled_prediction)
                        ha = 'right'
                    
                    ax.text(text_x, i, text, ha=ha, va='center', fontweight='bold', 
                           color='black', fontsize=10)
        
        # Set y-axis labels with better formatting
        ax.set_yticks(range(len(all_labels)))
        ax.set_yticklabels([])  # Remove default labels
        
        # Add custom y-axis labels
        for i, label in enumerate(all_labels):
            if i == 0 or i == len(all_labels)-1:
                # Bold for base value and final prediction
                ax.text(-0.01 * scaled_prediction, i, label, ha='right', va='center',
                       fontweight='bold', fontsize=11)
            else:
                ax.text(-0.01 * scaled_prediction, i, label, ha='right', va='center', fontsize=10)
        
        # Add current rent comparison line
        actual_rent = property_data.get('total_rent', 0)
        if actual_rent > 0 and abs(actual_rent - scaled_prediction) > 0.01 * scaled_prediction:
            ax.axvline(x=actual_rent, color='purple', linestyle='--', linewidth=2)
            # Position the current rent label at the top
            ax.text(actual_rent, -0.5, f"Current Rent: Rs. {actual_rent:,.0f}", 
                   color='purple', ha='center', va='bottom', fontweight='bold')
        
        # Set labels and title
        ax.set_title('Contributions to Rent Prediction Value', fontsize=16, pad=20)
        ax.set_xlabel('Rent Value (Rs.)', fontsize=12, labelpad=10)
        
        # Remove unnecessary spines
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Add subtle grid lines
        ax.xaxis.grid(True, linestyle='--', alpha=0.3)
        
        # Remove y-axis ticks
        ax.tick_params(axis='y', which='both', left=False)
        
        # Adjust layout and spacing
        plt.tight_layout()
        plt.subplots_adjust(left=0.2)  # Make room for labels
        
        return fig

    except Exception as e:
        print(f"Error in SHAP waterfall chart: {e}")
        import traceback
        traceback.print_exc()
        return dummy_waterfall_chart(property_data, target_prediction)

    #     import numpy as np
    #     import seaborn as sns
    #     sns.set_style("whitegrid")
    #     plt.figure(figsize=(12, 8))
        
    #     # Create lists for data - BASE FIRST, then features in descending impact, then FINAL
    #     labels = ['Base Value'] + features + ['Final Prediction']
        
    #     # Create the indices for proper vertical ordering (BASE at top)
    #     indices = list(range(len(labels)))
        
    #     # Create a blank DataFrame for the waterfall chart
    #     import pandas as pd
    #     chart_data = pd.DataFrame({
    #         'Label': labels,
    #         'Value': [0] * len(labels),
    #         'Start': [0] * len(labels),
    #         'End': [0] * len(labels),
    #         'Index': indices
    #     })
        
    #     # Add actual values - start with base value
    #     chart_data.loc[0, 'Value'] = scaled_base_value
    #     chart_data.loc[0, 'End'] = scaled_base_value
        
    #     # Add feature impacts
    #     running_total = scaled_base_value
    #     for i, impact in enumerate(scaled_impacts):
    #         idx = i + 1  # Skip the base value
    #         chart_data.loc[idx, 'Value'] = impact
    #         chart_data.loc[idx, 'Start'] = running_total
    #         running_total += impact
    #         chart_data.loc[idx, 'End'] = running_total
        
    #     # Add final prediction
    #     chart_data.loc[len(labels)-1, 'Value'] = scaled_prediction
    #     chart_data.loc[len(labels)-1, 'End'] = scaled_prediction
        
    #     # Create colors
    #     colors = []
    #     for i, row in chart_data.iterrows():
    #         if i == 0:  # Base value
    #             colors.append('#5DA5DA')  # Light blue
    #         elif i == len(labels) - 1:  # Final prediction
    #             colors.append('#000080')  # Navy blue
    #         elif row['Value'] >= 0:  # Positive impact
    #             colors.append('#60B05C')  # Green
    #         else:  # Negative impact
    #             colors.append('#D45C43')  # Red
        
    #     # Plot the chart
    #     fig, ax = plt.subplots(figsize=(12, max(8, len(labels) * 0.5)))
        
    #     # Plot the bars
    #     for i, row in chart_data.iterrows():
    #         if i == 0:  # Base value
    #             # Full bar for base value
    #             ax.barh(row['Index'], row['End'], left=0, color=colors[i], 
    #                    height=0.6, edgecolor='black', linewidth=0.5)
    #         elif i == len(labels) - 1:  # Final prediction
    #             # Full bar for final prediction
    #             ax.barh(row['Index'], row['End'], left=0, color=colors[i], 
    #                    height=0.6, edgecolor='black', linewidth=0.5)
    #         else:  # Feature impacts
    #             # Incremental bars for features
    #             value = row['Value']
    #             if value >= 0:  # Positive impact
    #                 ax.barh(row['Index'], value, left=row['Start'], color=colors[i], 
    #                        height=0.6, edgecolor='black', linewidth=0.5)
    #             else:  # Negative impact
    #                 ax.barh(row['Index'], -value, left=row['End'], color=colors[i], 
    #                        height=0.6, edgecolor='black', linewidth=0.5)
        
    #     # Add connecting lines
    #     for i in range(len(labels) - 1):
    #         if i == 0:  # From base to first feature
    #             start_x = chart_data.loc[i, 'End']
    #             ax.plot([start_x, start_x], [i+0.3, i+1-0.3], 'k--', linewidth=1, alpha=0.7)
    #         else:  # Between features
    #             if chart_data.loc[i, 'Value'] >= 0:
    #                 # Positive feature - connect from end
    #                 start_x = chart_data.loc[i, 'End']
    #             else:
    #                 # Negative feature - connect from start
    #                 start_x = chart_data.loc[i, 'Start']
    #             ax.plot([start_x, start_x], [i+0.3, i+1-0.3], 'k--', linewidth=1, alpha=0.7)
        
    #     # Add value labels
    #     for i, row in chart_data.iterrows():
    #         if i == 0:  # Base value
    #             # Label in the middle of the bar
    #             x_pos = row['End'] / 2
    #             text = f"Rs. {row['End']:,.0f}"
    #             color = 'white' if row['End'] > 30000 else 'black'
    #             ax.text(x_pos, i, text, ha='center', va='center', 
    #                    fontweight='bold', color=color, fontsize=10)
    #         elif i == len(labels) - 1:  # Final prediction
    #             # Label in the middle of the bar
    #             x_pos = row['End'] / 2
    #             text = f"Rs. {row['End']:,.0f}"
    #             ax.text(x_pos, i, text, ha='center', va='center', 
    #                    fontweight='bold', color='white', fontsize=10)
    #         else:  # Feature impacts
    #             value = row['Value']
    #             if abs(value) > 0.01 * scaled_prediction:  # Only label significant impacts
    #                 if value >= 0:
    #                     text = f"+Rs. {value:,.0f}"
    #                     x_pos = row['Start'] + value / 2
    #                 else:
    #                     text = f"Rs. {value:,.0f}"
    #                     x_pos = row['End'] + value / 2
                    
    #                 # Check if value is large enough for text to fit inside bar
    #                 if abs(value) > 0.1 * scaled_prediction:
    #                     ax.text(x_pos, i, text, ha='center', va='center', 
    #                            fontweight='bold', color='white' if abs(value) > 5000 else 'black', 
    #                            fontsize=9)
    #                 else:
    #                     # Place text to the left or right of the bar
    #                     if value >= 0:
    #                         ax.text(row['End'] + 0.01 * scaled_prediction, i, text, 
    #                                ha='left', va='center', fontweight='bold', fontsize=9)
    #                     else:
    #                         ax.text(row['Start'] - 0.01 * scaled_prediction, i, text, 
    #                                ha='right', va='center', fontweight='bold', fontsize=9)
        
    #     # Set y-ticks and labels
    #     ax.set_yticks(indices)
    #     ax.set_yticklabels(labels)
        
    #     # Add current rent line
    #     actual_rent = property_data.get('total_rent', 0)
    #     if actual_rent > 0:
    #         ax.axvline(x=actual_rent, color='purple', linestyle='--', linewidth=2)
    #         ax.text(actual_rent, -0.5, f"Current Rent: Rs. {actual_rent:,.0f}", 
    #                color='purple', ha='center', va='bottom', fontweight='bold')
        
    #     # Set title and axis labels
    #     ax.set_title('Contributions to Rent Prediction Value', fontsize=14, pad=20)
    #     ax.set_xlabel('Rent Value (Rs.)', fontsize=12)
        
    #     # Set axis limits
    #     max_value = max(chart_data['End'].max(), actual_rent if actual_rent > 0 else 0)
    #     ax.set_xlim(-0.05 * max_value, 1.1 * max_value)
        
    #     # Adjust layout
    #     plt.tight_layout()
        
    #     return fig
        
    # except Exception as e:
    #     print(f"Error in waterfall chart creation: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     return dummy_waterfall_chart(property_data, target_prediction)        
        
# def create_robust_shap_waterfall_chart(property_data, ml_model, feature_names, label_encoders=None,target_prediction=None):
#     """Create a robust waterfall chart with proper encoding of categorical variables"""
#     try:
#         # Basic validation
#         if ml_model is None or feature_names is None:
#             print("Model or feature names is None, using dummy chart")
#             return dummy_waterfall_chart(property_data, target_prediction)

#         # print(f"Creating SHAP waterfall chart with features: {feature_names}")

#         # Ensure we're only using features the model knows about
#         model_features = list(ml_model.feature_names_in_) if hasattr(ml_model, 'feature_names_in_') else feature_names
#         # print(f"Model features: {model_features}")

#         # Check for differences
#         extra_features = [f for f in feature_names if f not in model_features]
#         missing_features = [f for f in model_features if f not in feature_names]

#         # Use the correct feature list
#         feature_names = model_features

#         # Create a copy of property data
#         property_data_copy = property_data.copy()
#         input_df = pd.DataFrame([property_data_copy])

#         # Create model input with exact columns model expects
#         model_input = pd.DataFrame(index=[0], columns=feature_names)

#         # IMPORTANT: First process categorical features with proper encoding
#         categorical_features = ['furnishing', 'locality', 'society']
#         for feature in categorical_features:
#             if feature in feature_names:
#                 if feature in input_df.columns and label_encoders and feature in label_encoders:
#                     try:
#                         feature_value = input_df[feature].iloc[0]

#                         # Check if already numeric (pre-encoded)
#                         if isinstance(feature_value, (int, float, np.number)):
#                             model_input[feature] = feature_value
#                             # print(f"Using pre-encoded {feature}: {feature_value}")
#                         else:
#                             # Need to encode string value
#                             feature_str = str(feature_value)

#                             # Check if value exists in encoder classes
#                             if feature_str in label_encoders[feature].classes_:
#                                 encoded_value = label_encoders[feature].transform([feature_str])[0]
#                                 model_input[feature] = encoded_value
#                                 # print(f"Encoded {feature}: '{feature_str}' → {encoded_value}")
#                             else:
#                                 # Try case-insensitive match
#                                 feature_lower = feature_str.lower()
#                                 for i, cls in enumerate(label_encoders[feature].classes_):
#                                     if cls.lower() == feature_lower:
#                                         model_input[feature] = i
#                                         # print(f"Case-insensitive encoded {feature}: '{feature_str}' → {i} (class: '{cls}')")
#                                         break
#                                 else:
#                                     # No match found, use default (0)
#                                     model_input[feature] = 0
#                                     # print(f"No match for {feature}: '{feature_str}' in encoder, defaulting to 0")
#                     except Exception as e:
#                         print(f"Error encoding {feature}: {e}")
#                         model_input[feature] = 0
#                 else:
#                     model_input[feature] = 0
#                     print(f"Missing {feature} or encoder, defaulting to 0")

#         # Now process numeric features
#         for feature in feature_names:
#             if feature not in categorical_features:
#                 if feature == 'log_builtup_area':
#                     # Special handling for log_builtup_area
#                     if 'builtup_area' in input_df.columns:
#                         area_val = input_df['builtup_area'].iloc[0]
#                         model_input[feature] = np.log1p(area_val)
#                         # print(f"Calculated log_builtup_area: {model_input[feature].iloc[0]} from builtup_area: {area_val}")
#                     elif 'log_builtup_area' in input_df.columns:
#                         model_input[feature] = input_df['log_builtup_area'].iloc[0]
#                         # print(f"Using existing log_builtup_area: {model_input[feature].iloc[0]}")
#                     else:
#                         model_input[feature] = 0
#                         # print(f"Missing area data, defaulting log_builtup_area to 0")
#                 elif feature in input_df.columns:
#                     model_input[feature] = input_df[feature].iloc[0]
#                     # print(f"Using {feature}: {model_input[feature].iloc[0]}")
#                 else:
#                     # Try derived features
#                     if feature == 'floor_to_total_floors' and 'floor' in input_df.columns and 'total_floors' in input_df.columns:
#                         floor = input_df['floor'].iloc[0]
#                         total_floors = input_df['total_floors'].iloc[0]
#                         if total_floors > 0:
#                             model_input[feature] = floor / total_floors
#                             # print(f"Calculated {feature}: {model_input[feature].iloc[0]} from floor: {floor} and total_floors: {total_floors}")
#                         else:
#                             model_input[feature] = 0
#                             # print(f"Could not calculate {feature} (division by zero), defaulting to 0")
#                     elif feature == 'building_age' and 'building_age' not in input_df.columns:
#                         # Default to 5 years if not provided
#                         model_input[feature] = 5
#                         # print(f"Missing {feature}, using default value: 5")
#                     else:
#                         model_input[feature] = 0
#                         # print(f"Missing {feature}, defaulting to 0")

#         # Ensure all values are numeric
#         for col in model_input.columns:
#             if not pd.api.types.is_numeric_dtype(model_input[col]):
#                 try:
#                     model_input[col] = pd.to_numeric(model_input[col])
#                 except:
#                     # print(f"Warning: Could not convert {col} to numeric, defaulting to 0")
#                     model_input[col] = 0

#         # Verify we can make a prediction (debug only)
#         try:
#             direct_pred = ml_model.predict(model_input)[0]
#             # print(f"Direct model prediction: {direct_pred}")
#         except Exception as e:
#             print(f"Warning: Direct prediction failed: {e}")

#         # Create SHAP explainer
#         explainer = shap.TreeExplainer(ml_model)

#         # Get expected value
#         try:
#             base_value = explainer.expected_value
#             if hasattr(base_value, "__len__"):
#                 base_value = base_value[0]
#             # print(f"Base value: {base_value}")
#         except Exception as e:
#             # print(f"Could not get base value: {e}, using fallback")
#             base_value = property_data.get('total_rent', 25000)

#         # Calculate SHAP values
#         shap_values = explainer(model_input)

#         # Handle different SHAP library versions
#         if hasattr(shap_values, "values"):
#             shap_vals = shap_values.values[0]
#             feature_names_for_display = shap_values.feature_names
#         else:
#             shap_vals = shap_values[0] if hasattr(shap_values, "__len__") else shap_values
#             feature_names_for_display = feature_names

#         # Create DataFrames for plotting
#         waterfall_data = pd.DataFrame({
#             'feature': feature_names_for_display,
#             'shap_value': shap_vals
#         })

#         # Remove features with NaN or zero impact
#         waterfall_data = waterfall_data.dropna(subset=['shap_value'])
#         waterfall_data = waterfall_data[waterfall_data['shap_value'] != 0]

#         if len(waterfall_data) == 0:
#             print("No non-zero SHAP values, using dummy chart")
#             return dummy_waterfall_chart(property_data, target_prediction)

#         # Sort by absolute impact
#         waterfall_data['abs_shap'] = waterfall_data['shap_value'].abs()
#         waterfall_data = waterfall_data.sort_values('abs_shap', ascending=False)

#         # Format display names - Replace log_ and decode categorical values
#         display_names = []
#         for idx, row in waterfall_data.iterrows():
#             feature = row['feature']
#             disp_name = feature.replace('log_', '')

#             # Add decoded categorical values if available
#             if disp_name in categorical_features and label_encoders and disp_name in label_encoders:
#                 try:
#                     feature_value = model_input[feature].iloc[0]
#                     if hasattr(label_encoders[disp_name], 'inverse_transform'):
#                         original_value = label_encoders[disp_name].inverse_transform([int(feature_value)])[0]
#                         disp_name = f"{disp_name}: {original_value}"
#                 except Exception as e:
#                     print(f"Could not decode {disp_name}: {e}")

#             # Format name
#             disp_name = disp_name.replace('_', ' ').title()
#             display_names.append(disp_name)

#         waterfall_data['display_name'] = display_names

#         # Get feature and impact lists
#         features = waterfall_data['display_name'].tolist()
#         impacts = waterfall_data['shap_value'].tolist()

#         # Calculate total impact and prediction
#         total_impact = sum(impacts)
#         original_prediction = base_value + total_impact
#         # Calculate scaling factor if target prediction is provided
#         if target_prediction is not None and original_prediction > 0:
#             # Calculate scaling factor for SHAP values
#             scaling_factor = target_prediction / original_prediction
#             print(f"Scaling factor: {scaling_factor:.4f} (Target: {target_prediction:.2f} / Original: {original_prediction:.2f})")
            
#             # Scale the impacts and base value
#             scaled_impacts = [impact * scaling_factor for impact in impacts]
#             scaled_base_value = base_value * scaling_factor
#             scaled_prediction = scaled_base_value + sum(scaled_impacts)
            
#             print(f"Scaled base value: {scaled_base_value:.2f}")
#             print(f"Scaled prediction: {scaled_prediction:.2f} (Target: {target_prediction:.2f})")
#         else:
#             scaled_impacts = impacts
#             scaled_base_value = base_value
#             scaled_prediction = original_prediction


#         # Create figure
#         fig_height = max(8, len(features) * 0.4 + 2)
#         fig, ax = plt.subplots(figsize=(12, fig_height))

#         # Colors based on impact
#         colors = ['green' if impact > 0 else 'red' for impact in scaled_impacts]

#         # Plot the features
#         ax.barh(features, scaled_impacts, color=colors)

#         # Add base and total bars
#         ax.barh(['Base Value'], [scaled_base_value], color='gray')
#         ax.barh(['Final Prediction'], [scaled_prediction], color='blue')

#         # Add feature impact values
#         for i, impact in enumerate(scaled_impacts):
#             sign = "+" if impact > 0 else ""
#             ax.text(impact + (0.02 * max(abs(impact), 0.01)), i,
#                    f"{sign}Rs. {impact:,.0f}",
#                    va='center', fontsize=10)

#         # Add base and total values
#         ax.text(scaled_base_value * 1.02, len(features),
#                f"Rs. {scaled_base_value:,.0f}",
#                va='center', fontsize=10)

#         ax.text(scaled_prediction * 1.02, len(features) + 1,
#                f"Rs. {scaled_prediction:,.0f}",
#                va='center', fontsize=10, fontweight='bold')

#         # Set labels and title
#         ax.set_title('Feature Contribution to Rent Prediction', fontsize=14)
#         ax.set_xlabel('Impact on Rent Value (Rs. )', fontsize=12)

#         # Add actual rent for comparison
#         actual_rent = property_data.get('total_rent', 0)
#         if actual_rent > 0:
#             ax.axvline(x=actual_rent, color='purple', linestyle='--', linewidth=2,
#                       label=f'Actual Rent: Rs. {actual_rent:,.0f}')
#             ax.legend()

#         plt.grid(True, axis='x', alpha=0.3)
#         plt.tight_layout()

#         return fig

#     except Exception as e:
#         print(f"Error in SHAP waterfall chart: {e}")
#         import traceback
#         traceback.print_exc()
#         return dummy_waterfall_chart(property_data, target_prediction)

def create_feature_importance_fallback(property_data, ml_model, feature_names):
    # """Ultra-simple fallback visualization using direct feature importances"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get feature importances directly from model
        if hasattr(ml_model, 'feature_importances_'):
            # Clean feature names
            display_names = [f.replace('log_', '').replace('_', ' ').title()
                            for f in feature_names]

            # Sort importances
            importances = ml_model.feature_importances_
            if len(importances) > len(feature_names):
                importances = importances[:len(feature_names)]
            elif len(importances) < len(feature_names):
                # Pad with zeros
                importances = np.pad(importances, (0, len(feature_names) - len(importances)))

            # Sort features by importance
            sorted_indices = np.argsort(importances)[::-1]
            sorted_names = [display_names[i] for i in sorted_indices]
            sorted_importances = [importances[i] for i in sorted_indices]

            # Take top 8
            num_features = min(8, len(sorted_names))

            # Create horizontal bar chart
            ax.barh(sorted_names[:num_features], sorted_importances[:num_features], color='skyblue')

            ax.set_title('Feature Importance in Rent Prediction', fontsize=14)
            ax.set_xlabel('Relative Importance', fontsize=12)

            plt.tight_layout()
            return fig
        else:
            return dummy_waterfall_chart(property_data, target_prediction)

    except Exception as e:
        print(f"Even feature importance fallback failed: {e}")
        return dummy_waterfall_chart(property_data, target_prediction)

# Keep the dummy function as a fallback
def dummy_waterfall_chart(property_data, target_prediction = None):
   # """Create a dummy waterfall chart for feature impact"""
   plt.figure(figsize=(12, 8))

   # Define features and their impact
   features = ['builtup_area', 'locality', 'bedrooms', 'furnishing', 'floor', 'bathrooms']
   impacts = [9000, 6000, 4500, 3000, 2000, 1000]

   # Sort by impact
   sorted_idx = np.argsort(impacts)[::-1]
   features = [features[i] for i in sorted_idx]
   impacts = [impacts[i] for i in sorted_idx]

   # Base value
   base = 20000
       # Total from impacts
   total = base + sum(impacts)
    
   # Scale if target prediction provided
   if target_prediction is not None and total > 0:
       scaling_factor = target_prediction / total
       impacts = [impact * scaling_factor for impact in impacts]
       base = base * scaling_factor
       total = target_prediction  # Should equal base + sum(impacts)


   # Plot bars
   plt.barh(features, impacts, left=base, color='lightblue')

   # Plot base and total
   total = base + sum(impacts)
   plt.barh(['Base Value'], [base], color='gray')
   plt.barh(['Total Rent'], [total], color='green')

   # Add values
   for i, feature in enumerate(features):
       plt.text(base + impacts[i] + 100, i, f"+₹{impacts[i]:,}", va='center')

   plt.text(base + 100, len(features), f"₹{base:,}", va='center')
   plt.text(total + 100, len(features) + 1, f"₹{total:,}", va='center')

   plt.title('Feature Contribution to Rent Prediction (Simulated Data)', fontsize=14)
   plt.xlabel('Rent Value (₹)', fontsize=12)
   plt.tight_layout()

   return plt.gcf()

def generate_landlord_report(property_data, full_dataset, ml_model=None, feature_names=None, label_encoders=None, generate_plots=False, rent_estimates = None):
    # """
    # Generate a complete landlord report for a specific property with lazy-loaded visualizations

    # Parameters:
    # property_data: Series or dict with target property features
    # full_dataset: DataFrame with all properties
    # ml_model: Trained ML model for predictions (optional)
    # feature_names: List of feature names (optional)
    # label_encoders: Dictionary of label encoders (optional)
    # generate_plots: Whether to generate all plots immediately (default: False)

    # Returns:
    # dict: Complete report data and visualizations
    # """
    # Prepare data
    full_dataset = prepare_data(full_dataset)
    feature_names = [
        'bedrooms',
        'builtup_area',
        'bathrooms',
        'furnishing',
        'locality',
        'society',
        'floor',
        'total_floors',
        'floor_to_total_floors'
    ]

    if isinstance(property_data, dict):
        # Convert property data to a format for analysis
        property_series = pd.Series(property_data)

        # Calculate derived metrics if needed
        if 'rent_per_sqft' not in property_data:
            property_data['rent_per_sqft'] = property_data['total_rent'] / property_data['builtup_area']

        if 'floor_to_total_floors' not in property_data:
            property_data['floor_to_total_floors'] = property_data['floor'] / property_data['total_floors']
    else:
        property_series = property_data
        property_data = property_series.to_dict()


    if rent_estimates:
        # Calculate combined estimate (average of model B and sqft method)
        combined_estimate = (rent_estimates.get('model_b', 0) + rent_estimates.get('sqft_method', 0)) / 2

    # Find comparable properties using the hierarchical approach
    comparables_dict = find_comparable_properties(property_data, full_dataset)

    # Determine primary comparables for analysis
    primary_group = None
    primary_comparables = None
    comp_groups = ['same_society_same_bhk', 'same_locality_same_bhk',
                  'same_society_diff_bhk', 'same_locality_similar_bhk']

    for group in comp_groups:
        if group in comparables_dict and len(comparables_dict[group]) > 0:
            primary_comparables = comparables_dict[group]
            primary_group = group
            break

    if primary_comparables is None or len(primary_comparables) == 0:
        primary_comparables = comparables_dict.get('top_comparables', pd.DataFrame())
        primary_group = 'all_comparables'

    # Calculate market position
    market_position = calculate_market_position(property_data, comparables_dict, full_dataset)


    # Store visualization functions for deferred execution (lazy loading)
    visualization_generators = {
        'position_chart': lambda: create_property_position_chart(property_data, comparables_dict),
        'feature_radar': lambda: create_feature_comparison_radar(property_data, comparables_dict, full_dataset),
        'feature_impact': lambda: create_robust_shap_waterfall_chart(property_data, ml_model, feature_names, label_encoders, combined_estimate),
        'rent_distribution': lambda: plot_rent_distribution_with_property(property_data, full_dataset)
    }

    # Generate plots immediately if requested, otherwise leave them for lazy loading
    visualizations = {}
    if generate_plots:
        print("Generating all visualizations upfront...")
        for name, generator in visualization_generators.items():
            try:
                visualizations[name] = generator()
                print(f"  Generated {name} visualization")
            except Exception as e:
                print(f"  Error generating {name} visualization: {e}")

    # Helper function to analyze each tier of comparables
    def analyze_comparable_tier(tier_df, property_data):
        """Analyze a specific tier of comparable properties"""
        if len(tier_df) == 0:
            return {
                'count': 0,
                'available': False
            }

        property_rent = property_data.get('total_rent')
        property_rent_psf = property_data.get('rent_per_sqft')

        # Calculate premium/discount based on rent per square foot
        avg_rent_psf = tier_df['rent_per_sqft'].mean() if 'rent_per_sqft' in tier_df.columns else 0
        premium_discount_psf = ((property_rent_psf - avg_rent_psf) / avg_rent_psf) * 100 if avg_rent_psf > 0 else 0

        # Calculate premium/discount based on absolute rent (for reference)
        avg_rent = tier_df['total_rent'].mean()
        premium_discount_absolute = ((property_rent - avg_rent) / avg_rent) * 100 if avg_rent > 0 else 0

        return {
            'count': len(tier_df),
            'available': True,
            'avg_rent': tier_df['total_rent'].mean(),
            'median_rent': tier_df['total_rent'].median(),
            'min_rent': tier_df['total_rent'].min(),
            'max_rent': tier_df['total_rent'].max(),
            'avg_rent_psf': avg_rent_psf,
            'premium_discount_psf': premium_discount_psf,
            'premium_discount': premium_discount_absolute,
            'percentile': percentileofscore(tier_df['total_rent'], property_rent) if len(tier_df) > 0 else None,
            'percentile_psf': percentileofscore(tier_df['rent_per_sqft'], property_rent_psf) if 'rent_per_sqft' in tier_df.columns and len(tier_df) > 0 else None
        }
    # Modified version to preserve string values
    # Store the original values from the input property_data
    original_society = property_data.get('society')
    original_locality = property_data.get('locality')

    # If these are integers, get the actual names if available
    society_display = original_society
    locality_display = original_locality

    if isinstance(original_society, (int, float)) and label_encoders and 'society' in label_encoders:
        try:
            society_idx = int(original_society)
            if hasattr(label_encoders['society'], 'classes_') and 0 <= society_idx < len(label_encoders['society'].classes_):
                society_display = label_encoders['society'].classes_[society_idx]
        except:
            pass

    if isinstance(original_locality, (int, float)) and label_encoders and 'locality' in label_encoders:
        try:
            locality_idx = int(original_locality)
            if hasattr(label_encoders['locality'], 'classes_') and 0 <= locality_idx < len(label_encoders['locality'].classes_):
                locality_display = label_encoders['locality'].classes_[locality_idx]
        except:
            pass

    # Compile the full report with enhanced comparables structure
    report = {
        'property_id': property_data.get('property_id', 'Unknown'),
        'report_date': datetime.now().strftime('%Y-%m-%d'),
        'property_details': {
            'location': {
                'society': society_display,
                'locality': locality_display
            },
            'physical': {
                'bedrooms': property_data.get('bedrooms'),
                'bathrooms': property_data.get('bathrooms'),
                'builtup_area': property_data.get('builtup_area')
            },
            'building': {
                'floor': property_data.get('floor'),
                'total_floors': property_data.get('total_floors'),
                'floor_to_total_floors': property_data.get('floor_to_total_floors')
            },
            'condition': {
                'furnishing': property_data.get('furnishing')
            },
            'pricing': {
                'total_rent': property_data.get('total_rent'),
                'rent_per_sqft': property_data.get('rent_per_sqft')
            }
        },
        'market_position': market_position,
        # 'feature_impact': feature_impact,
        'comparables': {
            'primary_group': market_position.get('primary_comparison_group', primary_group),
            'tiered_analysis': {
                'same_society_same_bhk': analyze_comparable_tier(comparables_dict.get('same_society_same_bhk', pd.DataFrame()), property_data),
                'same_locality_same_bhk': analyze_comparable_tier(comparables_dict.get('same_locality_same_bhk', pd.DataFrame()), property_data),
                'same_society_diff_bhk': analyze_comparable_tier(comparables_dict.get('same_society_diff_bhk', pd.DataFrame()), property_data),
                'same_locality_similar_bhk': analyze_comparable_tier(comparables_dict.get('same_locality_similar_bhk', pd.DataFrame()), property_data)
            },
            'all_comparables': comparables_dict.get('top_comparables', pd.DataFrame())
        },
        'visualizations': visualizations,  # Pre-generated visualizations (if any)
        'visualization_generators': visualization_generators,  # Functions to generate visualizations on demand
        'summary': {
            'current_rent': property_data.get('total_rent'),
            'market_position': market_position['position_category'],
            'premium_discount': {
                'vs_society': market_position['premium_discount'].get('same_society_same_bhk_avg',
                                                                     market_position['premium_discount'].get('locality_avg', 0)),
                'vs_locality': market_position['premium_discount'].get('same_locality_same_bhk_avg',
                                                                     market_position['premium_discount'].get('locality_avg', 0)),
                'overall': market_position['premium_discount'].get('locality_avg', 0)
            },
            # 'top_recommendations': recommendations[:3] if len(recommendations) >= 3 else recommendations,
            # 'potential_increase': sum([r.get('potential_impact', 0) for r in recommendations]) / 100 * property_data.get('total_rent')
        }
    }

        # Near the end of the generate_landlord_report function, before returning the report:
    if rent_estimates:
        report['rent_estimates'] = {
            'model_a': rent_estimates.get('model_a', 0),
            'model_b': rent_estimates.get('model_b', 0),
            'sqft_method': rent_estimates.get('sqft_method', 0),
            'combined_estimate': combined_estimate,
            'lower_bound': combined_estimate * 0.95,  # -5%
            'upper_bound': combined_estimate * 1.05   # +5%
        }

    if label_encoders and 'society' in label_encoders and 'locality' in label_encoders:
        # Create mappings from encoded values to original names
        society_name_map = {}
        locality_name_map = {}

        # Create society mapping
        if hasattr(label_encoders['society'], 'classes_'):
            for idx, name in enumerate(label_encoders['society'].classes_):
                society_name_map[str(idx)] = name

        # Create locality mapping
        if hasattr(label_encoders['locality'], 'classes_'):
            for idx, name in enumerate(label_encoders['locality'].classes_):
                locality_name_map[str(idx)] = name

        # Add to report
        if 'metadata' not in report:
            report['metadata'] = {}

        report['metadata']['society_name_map'] = society_name_map
        report['metadata']['locality_name_map'] = locality_name_map
                # Add at the end of generate_landlord_report before returning
        print("\nDEBUG - Report Data Structure:")
        print(f"Society in report: {report.get('property_details', {}).get('location', {}).get('society')}")
        print(f"Locality in report: {report.get('property_details', {}).get('location', {}).get('locality')}")
    return report

def show_full_report_data(report):
    # """
    # Display the complete data structure of a landlord report
    # to help identify what's available for the display function.

    # Parameters:
    # report: The dictionary returned by generate_landlord_report
    # """
    import json
    from pprint import pprint

    # Helper function to make the output more readable
    def format_value(value):
        if isinstance(value, float):
            if value > 1000:
                return f"{value:,.2f}"
            return f"{value:.2f}"
        elif isinstance(value, (dict, list)) and not value:
            return "Empty"
        return value

    def print_section(title, data, indent=0):
        spaces = ' ' * indent
        print(f"{spaces}{title}:")

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)) and value:
                    print(f"{spaces}  {key}:")
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, (dict, list)) and subvalue:
                                print_section(subkey, subvalue, indent + 4)
                            else:
                                formatted = format_value(subvalue)
                                print(f"{spaces}    {subkey}: {formatted}")
                    else:  # list
                        if all(isinstance(item, (int, float, str, bool, type(None))) for item in value):
                            print(f"{spaces}    {value}")
                        else:
                            for i, item in enumerate(value):
                                print(f"{spaces}    Item {i+1}:")
                                if isinstance(item, dict):
                                    for ikey, ivalue in item.items():
                                        formatted = format_value(ivalue)
                                        print(f"{spaces}      {ikey}: {formatted}")
                                else:
                                    print(f"{spaces}      {item}")
                else:
                    formatted = format_value(value)
                    print(f"{spaces}  {key}: {formatted}")
        else:
            print(f"{spaces}  {data}")

    # Print each main section of the report
    print("\n==== FULL LANDLORD REPORT DATA STRUCTURE ====\n")

    print("1. BASIC INFORMATION:")
    print(f"  Property ID: {report.get('property_id', 'Unknown')}")
    print(f"  Report Date: {report.get('report_date', 'Unknown')}")

    # Property Details
    print_section("2. PROPERTY DETAILS", report.get('property_details', {}))

    # Market Position
    print("\n3. MARKET POSITION:")
    market_position = report.get('market_position', {})
    print(f"  Position Category: {market_position.get('position_category', 'Unknown')}")
    print(f"  Primary Comparison Group: {market_position.get('primary_comparison_group', 'Unknown')}")
    print(f"  Number of Primary Comparables: {market_position.get('num_primary_comparables', 0)}")

    print("\n  3.1 PERCENTILE RANKS:")
    percentile_ranks = market_position.get('percentile_ranks', {})
    for group, ranks in percentile_ranks.items():
        print(f"    {group}:")
        for metric, value in ranks.items():
            print(f"      {metric}: {value:.1f}" if isinstance(value, float) else f"      {metric}: {value}")

    print("\n  3.2 PREMIUM/DISCOUNT (%):")
    premium_discount = market_position.get('premium_discount', {})
    for group, value in premium_discount.items():
        print(f"    {group}: {value:.2f}%" if isinstance(value, float) else f"    {group}: {value}")

    # Feature Impact
    print("\n4. FEATURE IMPACT:")
    feature_impact = report.get('feature_impact', {})
    for feature, impact in feature_impact.items():
        print(f"  {feature}:")
        for metric, value in impact.items():
            if isinstance(value, float):
                if metric == 'importance':
                    print(f"    {metric}: {value*100:.1f}%")
                else:
                    print(f"    {metric}: {value:.2f}")
            else:
                print(f"    {metric}: {value}")

    # Comparables
    print("\n5. COMPARABLES:")
    comparables = report.get('comparables', {})
    print(f"  Primary Group: {comparables.get('primary_group', 'Unknown')}")

    print("\n  5.1 TIERED ANALYSIS:")
    tiered_analysis = comparables.get('tiered_analysis', {})
    for tier, analysis in tiered_analysis.items():
        print(f"    {tier}:")
        if isinstance(analysis, dict):
            for key, value in analysis.items():
                formatted = format_value(value)
                print(f"      {key}: {formatted}")

    # Summary
    print("\n6. SUMMARY:")
    summary = report.get('summary', {})
    for key, value in summary.items():
        if key == 'premium_discount':
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                formatted = format_value(subvalue)
                print(f"    {subkey}: {formatted}")
        elif key == 'top_recommendations':
            print(f"  {key}: {len(value)} recommendations")
        else:
            formatted = format_value(value)
            print(f"  {key}: {formatted}")

    # Visualizations (just list them, not showing)
    print("\n7. VISUALIZATIONS:")
    visualizations = report.get('visualizations', {})
    for viz_name in visualizations.keys():
        print(f"  {viz_name}")

    print("\n==== END OF REPORT DATA STRUCTURE ====")

def display_landlord_report_visuals(report, max_charts=2, selected_charts=None):
    # """
    # Display a curated selection of visualizations from the landlord report

    # Parameters:
    # report: The landlord report dictionary
    # max_charts: Maximum number of charts to display (default: 2)
    # selected_charts: List of specific chart names to display (optional)
    # """
    # Get already generated visualizations
    visualizations = report.get('visualizations', {})
    # Get visualization generators for lazy loading
    visualization_generators = report.get('visualization_generators', {})

    if not visualizations and not visualization_generators:
        print("No visualizations available in the report")
        return

    # Define priority order for visualizations
    priority_order = [
        'position_chart',       # Most important - shows property vs comparables
        'feature_radar',        # Good overview of property features
        'feature_impact',       # Shows what drives value
        'rent_distribution'     # Shows broader market context
    ]

    available_charts = list(visualizations.keys()) + [name for name in visualization_generators.keys()
                                                     if name not in visualizations]
    # print(f"Available charts: {available_charts}")

    # Determine which charts to display
    charts_to_display = []

    # Filter to selected charts if specified
    if selected_charts:
        # Only include charts that are available
        charts_to_display = [name for name in selected_charts if name in available_charts]
        # print(f"Charts being displayed: {charts_to_display}")
    else:
        # Sort by priority and limit to max_charts
        for name in priority_order:
            if name in available_charts and len(charts_to_display) < max_charts:
                charts_to_display.append(name)
        # print(f"Charts being displayed: {charts_to_display}")

    # Check if we have charts to display
    if not charts_to_display:
        print("No visualizations selected")
        return

    # Create a descriptive title lookup
    chart_titles = {
        'position_chart': 'Your Property vs. Comparable Properties',
        'feature_radar': 'Property Feature Comparison',
        'feature_impact': 'Factors Affecting Rent Value',
        'rent_distribution': 'Market Rent Distribution'
    }

    for i, name in enumerate(charts_to_display, 1):
        # Get or generate the figure
        if name in visualizations:
            fig = visualizations[name]
            print(f"Using pre-generated {name} visualization")
        else:
            # Generate the plot on-demand if not already created
            # print(f"Generating {name} visualization on demand...")
            try:
                fig = visualization_generators[name]()
                # Store for future use
                visualizations[name] = fig
            except Exception as e:
                print(f"Error generating {name} visualization: {e}")
                continue

        # Activate this figure for display
        plt.figure(fig.number)
        plt.tight_layout()

        title = chart_titles.get(name, name.replace('_', ' ').title())
        print(f"\nChart {i}: {title}")

        # Show the chart
        # plt.show()
        return plt.gcf()

        # Add a brief explanation based on chart type
        if name == 'position_chart':
            position = report['market_position']['position_category']
            print(f"This chart shows how your property's rent (red line) compares to similar properties.")
            print(f"Your property is positioned in the '{position}' segment of the market.")

        elif name == 'feature_radar':
            print("This radar chart compares your property's key features to comparable properties.")
            print("Areas where your property's line (red) extends further than others indicate strengths.")

        elif name == 'feature_impact':
            print("This chart shows how different features contribute to your property's rent value.")
            print("Green bars indicate positive contributions, red bars indicate negative impacts.")

        elif name == 'rent_distribution':
            percentile = report['market_position']['percentile_ranks'].get('overall_market', {}).get('rent', 0)
            print(f"This chart shows your property's position in the overall market rent distribution.")
            print(f"Your property is at the {percentile:.0f}th percentile of the market.")

def safe_format_categorical(value):
    # """
    # Format categorical value with type safety - handles any type without errors

    # Parameters:
    # value: The value to format (can be string, int, float, etc.)

    # Returns:
    # str: Safely formatted string representation of the value
    # """
    import numpy as np  # Ensure numpy is imported

    if isinstance(value, str):
        # For strings, capitalize each word
        return value.title()
    elif isinstance(value, (int, float, np.number)):
        # For numeric values, add "(encoded)" to indicate it's an encoded value
        return f"{value} (encoded)"
    elif value is None:
        # For None values
        return "Not specified"
    else:
        # For any other type, convert to string
        return str(value)


def display_landlord_report(report, chart_selection=None):
    """
    Display the landlord report in a structured format with enhanced comparable analysis

    Parameters:
    report: The landlord report dictionary
    chart_selection: List of chart types to display (optional)
    """
    print("\n" + "="*50)
    print(f"PROPERTY RENTAL ANALYSIS REPORT")
    print(f"Generated on: {report['report_date']}")
    print("="*50)

    # Property details
    details = report['property_details']
    print("\n📍 PROPERTY DETAILS:")
    print(f"Location: {details['location']['locality']} | {details['location']['society']}")
    print(f"Configuration: {details['physical']['bedrooms']} BHK, {details['physical']['bathrooms']} Bathrooms")
    print(f"Area: {details['physical']['builtup_area']} sq.ft.")
    print(f"Floor: {details['building']['floor']} out of {details['building']['total_floors']}")
    # Use safe formatting for furnishing
    furnishing_display = safe_format_categorical(details['condition']['furnishing'])
    print(f"Furnishing: {furnishing_display}")
    print(f"Current Rent: ₹{details['pricing']['total_rent']:,} (₹{details['pricing']['rent_per_sqft']:.2f}/sq.ft)")

    # Market position - enhanced with comparable tiers
    market = report['market_position']
    comparables = report['comparables']
    primary_group = comparables['primary_group']

    print("\n📊 MARKET POSITION:")
    print(f"Assessment: {market['position_category']}")
    print(f"Based on: {primary_group.replace('_', ' ').title()} comparison")

    # Premium/discount at different levels
    premium_discount = report['summary']['premium_discount']
    print("\nPREMIUM/DISCOUNT ANALYSIS:")

    # Society-level comparison
    society_tier = comparables['tiered_analysis']['same_society_same_bhk']
    if society_tier.get('available', False):
        society_premium = society_tier.get('premium_discount', 0)
        premium_text = f"{abs(society_premium):.1f}% {'ABOVE' if society_premium > 0 else 'BELOW'}"
        print(f"• Same Society, Same BHK: {premium_text} (based on {society_tier['count']} properties)")

    # Locality-level comparison
    locality_tier = comparables['tiered_analysis']['same_locality_same_bhk']
    if locality_tier.get('available', False):
        locality_premium = locality_tier.get('premium_discount', 0)
        premium_text = f"{abs(locality_premium):.1f}% {'ABOVE' if locality_premium > 0 else 'BELOW'}"
        print(f"• Same Locality, Same BHK: {premium_text} (based on {locality_tier['count']} properties)")

    # Overall comparison
    overall_premium = premium_discount.get('overall', 0)
    premium_text = f"{abs(overall_premium):.1f}% {'ABOVE' if overall_premium > 0 else 'BELOW'}"
    print(f"• Overall Market: {premium_text}")

    # Percentile rankings - enhanced with tiered analysis
    percentiles = market['percentile_ranks']
    print(f"\nRENT PERCENTILES:")

    # Show percentiles for each available tier
    for tier_name, tier_data in comparables['tiered_analysis'].items():
        if tier_data.get('available', False) and tier_data.get('count', 0) >= 3:
            # Only show if we have enough data
            percentile = tier_data.get('percentile')
            if percentile is not None:
                print(f"• {tier_name.replace('_', ' ').title()}: {percentile:.1f}th percentile")

    # Always show overall market percentile
    if 'overall_market' in percentiles:
        print(f"• Overall Market: {percentiles['overall_market'].get('rent', 50):.1f}th percentile")

    # Comparables summary - enhanced with tiered details
    print("\n🏘️ COMPARABLE PROPERTIES ANALYSIS:")

    # Show stats for each tier with sufficient data
    for tier_name, tier_data in comparables['tiered_analysis'].items():
        if tier_data.get('available', False) and tier_data.get('count', 0) >= 3:
            tier_display_name = tier_name.replace('_', ' ').title()
            print(f"\n• {tier_display_name} ({tier_data['count']} properties):")
            print(f"  Average Rent: ₹{tier_data['avg_rent']:,.2f}")
            print(f"  Rent Range: ₹{tier_data['min_rent']:,.2f} - ₹{tier_data['max_rent']:,.2f}")

            # Show premium/discount for this tier
            premium = tier_data.get('premium_discount', 0)
            if abs(premium) > 0.5:  # Only show if meaningful difference
                status = "above" if premium > 0 else "below"
                print(f"  Your property is {abs(premium):.1f}% {status} the average in this group")

    # Summary with enhanced insights
    print("\n📑 SUMMARY:")
    print(f"Current Rent: ₹{report['summary']['current_rent']:,}")
    print(f"Market Position: {report['summary']['market_position']}")

    # Add location-specific insights
    society_analysis = comparables['tiered_analysis']['same_society_same_bhk']
    locality_analysis = comparables['tiered_analysis']['same_locality_same_bhk']

    if society_analysis.get('available', False) and locality_analysis.get('available', False):
        society_premium = society_analysis.get('premium_discount', 0)
        locality_premium = locality_analysis.get('premium_discount', 0)

        if abs(society_premium - locality_premium) > 5:
            # Significant difference between society and locality positioning
            if society_premium > locality_premium:
                print(f"Your property is positioned higher within your society than in the broader locality")
            else:
                print(f"Your property is positioned higher in the locality market than within your society")

    # Display visualizations
    print("\n📈 VISUALIZATIONS:")
    # Use the chart_selection parameter if provided, otherwise default to feature_radar
    selected_charts = chart_selection if chart_selection else ['feature_radar']
    display_landlord_report_visuals(report, selected_charts=selected_charts)

    print("\n" + "="*50)
    print("End of Report")
    print("="*50)

def encode_property_for_analysis(property_data, df, label_encoders=None):
    # """
    # Encode property data to match dataset format and prepare for analysis

    # Parameters:
    # property_data: Dict with property features (original format)
    # df: DataFrame with all properties
    # label_encoders: Dictionary of label encoders (optional)
    # __add
    # Returns:
    # Dict: Property data with values encoded to match dataset format
    # """
    # Create a copy to avoid modifying the original
    encoded_property = property_data.copy()

    # print("\n=== Encoding Property Data ===")

    # Handle specific categorical features that need encoding
    categorical_cols = ['society', 'locality', 'furnishing']

    for col in categorical_cols:
        if col in encoded_property and col in df.columns:
            original_value = encoded_property[col]
            # print(f"Processing {col}: Original value = '{original_value}' ({type(original_value).__name__})")

            # 1. First check if dataset column is numeric but property value is string
            if pd.api.types.is_numeric_dtype(df[col]) and isinstance(original_value, str):
                try:
                    # Try direct conversion to number
                    numeric_value = float(original_value) if '.' in original_value else int(original_value)
                    encoded_property[col] = numeric_value
                    # print(f"  Converted string to number: '{original_value}' → {numeric_value}")
                except (ValueError, TypeError):
                    # If can't convert directly, try to find a numeric match in the dataset
                    # print(f"  Cannot directly convert '{original_value}' to number, checking label encoders")

                    # Try label encoder if available
                    if label_encoders and col in label_encoders:
                        try:
                            if original_value in label_encoders[col].classes_:
                                encoded_value = label_encoders[col].transform([original_value])[0]
                                encoded_property[col] = encoded_value
                                # print(f"  Encoded via label encoder: '{original_value}' → {encoded_value}")
                            else:
                                # print(f"  Warning: '{original_value}' not found in label encoder classes")
                                # Use most common value as fallback
                                most_common = df[col].value_counts().index[0]
                                encoded_property[col] = most_common
                                # print(f"  Using most common value as fallback: {most_common}")
                        except Exception as e:
                            # print(f"  Error using label encoder: {e}")
                            # Fallback to most common value
                            most_common = df[col].value_counts().index[0]
                            encoded_property[col] = most_common
                            # print(f"  Using most common value as fallback: {most_common}")
                    else:
                        # No label encoder, use most common value
                        most_common = df[col].value_counts().index[0]
                        encoded_property[col] = most_common
                        # print(f"  No label encoder available, using most common value: {most_common}")

            # 2. Handle case where dataset column is string but property value is numeric
            elif df[col].dtype == 'object' and isinstance(original_value, (int, float)):
                # Convert to string
                str_value = str(original_value)

                # Check if this string exists in the dataset
                if str_value in df[col].values:
                    encoded_property[col] = str_value
                    # print(f"  Converted number to string: {original_value} → '{str_value}'")
                else:
                    # Try label encoder in reverse direction
                    if label_encoders and col in label_encoders:
                        try:
                            # Find class that corresponds to this numeric value
                            for i, cls in enumerate(label_encoders[col].classes_):
                                if i == original_value:
                                    encoded_property[col] = cls
                                    # print(f"  Decoded via label encoder: {original_value} → '{cls}'")
                                    break
                            else:
                                # If no match, use the string version anyway
                                encoded_property[col] = str_value
                                # print(f"  No matching class found, using string version: '{str_value}'")
                        except Exception as e:
                            # print(f"  Error decoding via label encoder: {e}")
                            encoded_property[col] = str_value
                    else:
                        # No label encoder, just use string version
                        encoded_property[col] = str_value
                        # print(f"  No label encoder available, using string version: '{str_value}'")

            # 3. Handle furnishing specifically (common format issues)
            if col == 'furnishing' and isinstance(encoded_property[col], str):
                # Normalize furnishing format
                furnishing_map = {
                    'unfurnished': ['unfurnished', 'un furnished', 'un-furnished', 'not furnished'],
                    'semi furnished': ['semifurnished', 'semi furnished', 'semi-furnished', 'partially furnished'],
                    'furnished': ['furnished', 'fully furnished', 'full furnished']
                }

                # Get the original value after any conversions above
                furnishing_value = encoded_property[col].lower()

                # Find the standard format
                for standard, variations in furnishing_map.items():
                    if any(var == furnishing_value or var in furnishing_value for var in variations):
                        # Check if this standard exists in dataset
                        matching_values = df[df[col].str.lower().str.contains(standard, na=False)][col].unique() if df[col].dtype == 'object' else []

                        if len(matching_values) > 0:
                            encoded_property[col] = matching_values[0]
                            # print(f"  Normalized furnishing: '{original_value}' → '{matching_values[0]}'")
                        else:
                            encoded_property[col] = standard
                            # print(f"  Standardized furnishing: '{original_value}' → '{standard}'")
                        break

    # Always ensure rent_per_sqft is calculated
    if 'total_rent' in encoded_property and 'builtup_area' in encoded_property:
        if 'rent_per_sqft' not in encoded_property or encoded_property['rent_per_sqft'] is None:
            encoded_property['rent_per_sqft'] = encoded_property['total_rent'] / encoded_property['builtup_area']
            # print(f"  Calculated rent_per_sqft: {encoded_property['rent_per_sqft']:.2f}")

    # Always ensure floor_to_total_floors is calculated
    if 'floor' in encoded_property and 'total_floors' in encoded_property:
        if 'floor_to_total_floors' not in encoded_property or encoded_property['floor_to_total_floors'] is None:
            encoded_property['floor_to_total_floors'] = encoded_property['floor'] / encoded_property['total_floors']
            # print(f"  Calculated floor_to_total_floors: {encoded_property['floor_to_total_floors']:.2f}")

    # print("=== Encoding Complete ===")

    return encoded_property

def main(df, ml_model=None, label_encoders=None, feature_names=None):
    # """
    # Main function to run the entire analysis

    # Parameters:
    # df: DataFrame with rental property data
    # ml_model: Trained ML model for predictions (from another Colab cell)
    # label_encoders: Dictionary of label encoders (from another Colab cell)
    # feature_names: List of feature names used by the model (from another Colab cell)
    # """
    # Prepare the dataset
    df = prepare_data(df)

    print(f"Data loaded successfully. {len(df)} records found.")

    # Define sample property for analysis
    custom_property = {
        'property_id': 'CUSTOM001',
        'locality': 'Gachibowli',
        'society': 'Prestige High Fields',
        'bedrooms': 3.5,
        'bathrooms': 3,
        'builtup_area': 1995,
        # 'log_builtup_area': np.log1p(1995),  # Pre-calculated log
        'floor': 15,
        'total_floors': 35,
        'furnishing': 'Semi-Furnished',
        'total_rent': 85000,
        'rent_per_sqft': 85000/1995,
        'building_age': 8
    }
    encoded_property = encode_property_for_analysis(custom_property, df, label_encoders)

    # Use the custom property for this demo
    target_property = custom_property

    society_locality_map = build_society_locality_map(df, label_encoders)

    consistent_results = predict_rent_with_canonical_locality(target_property, society_locality_map)
    print(f"\nConsistent Prediction (using canonical locality):")
    print(f"- Model A (Raw Rent): ₹{consistent_results['model_a_raw_prediction']:.0f}")
    print(f"- Model B (Log Rent): ₹{consistent_results['model_b_log_prediction']:.0f}")

    estimated_rent = estimate_rent_alternative(
        area=target_property['builtup_area'],
        locality=target_property['locality'],
        society=target_property['society'],
        furnishing=target_property['furnishing']
    )
    print(f"💰 Rent Estimated via Rent/sqft: ₹{estimated_rent:,.0f}")


    # Print basic information about the target property
    print(f"\nAnalyzing property: {target_property['property_id']}")
    print(f"Location: {target_property['locality']}, {target_property['society']}")
    print(f"Configuration: {target_property['bedrooms']} BHK, {target_property['bathrooms']} bathrooms")
    print(f"Area: {target_property['builtup_area']} sq.ft., Floor: {target_property['floor']}/{target_property['total_floors']}")
    print(f"Furnishing: {target_property['furnishing']}")
    print(f"Current Rent: Rs.{target_property['total_rent']:,}")

    # Check if ML model and label encoders are available
    if ml_model is None:
        print("\nWarning: ML model not provided. Feature impact analysis will use dummy data.")
    if label_encoders is None:
        print("\nWarning: Label encoders not provided. Feature encoding might be limited.")
    if feature_names is None:
        print("\nWarning: Feature names not provided. Using default feature list.")
        feature_names = [
            'bedrooms',
            'builtup_area',
            'bathrooms',
            'furnishing',
            'locality',
            'society',
            'floor',
            'total_floors',
            'floor_to_total_floors'
        ]

    print("\nDEBUG - Property Location Values:")
    print(f"Society: {target_property['society']} ({type(target_property['society']).__name__})")
    print(f"Locality: {target_property['locality']} ({type(target_property['locality']).__name__})")


    # Generate the landlord report - set generate_plots=False for lazy loading
    print("\nGenerating comprehensive landlord report...")
    landlord_report = generate_landlord_report(
        encoded_property,
        df,
        ml_model=ml_model,
        feature_names=feature_names,
        label_encoders=label_encoders,
        generate_plots=False,
        rent_estimates={
        'model_a': consistent_results['model_a_raw_prediction'],
        'model_b': consistent_results['model_b_log_prediction'],
        'sqft_method': estimated_rent
    }
    )
    pdf_path = create_landlord_pdf_report (landlord_report, label_encoders=label_encoders,output_dir="landlord_reports")
    # Show report data structure for reference
    # show_full_report_data(landlord_report)

    # Display the report with specific chart selection
    charts_to_display = ['position_chart', 'feature_radar', 'feature_impact', 'rent_distribution']
    # charts_to_display = ['feature_radar']  # Just one chart
    # charts_to_display = None  # Show default charts by priority
    display_landlord_report(landlord_report, chart_selection=charts_to_display)



    # Provide the landlord with actionable insights
    print("\nKey Insights and Recommendations:")

    # Pricing insight
    position = landlord_report['market_position']['position_category']
    primary_group = landlord_report['market_position'].get('primary_comparison_group', 'overall_market')
    if primary_group + '_avg' in landlord_report['market_position']['premium_discount']:
        premium = landlord_report['market_position']['premium_discount'][primary_group + '_avg']
    else:
        # Fallback to locality average if specific group premium is not available
        premium = landlord_report['market_position']['premium_discount'].get('locality_avg', 0)


    if position in ["Premium", "Above Market"]:
        print(f"• Your property commands a {premium:.1f}% premium over comparable properties")
        if premium > 15:
            print("  Consider ensuring amenities and condition justify this premium to maintain occupancy")
    elif position in ["Below Market", "Significantly Below Market"]:
        print(f"• Your property is priced {abs(premium):.1f}% below comparable properties")
        print(f"  Consider a gradual rent increase to align closer with market rates")
    else:
        print("• Your property is priced appropriately for the market")

    print("\nReport generation complete!")


# if __name__ == "__main__":
#     ml_model = model_a
#     features_with_log = [
#     'bedrooms',
#     'log_builtup_area',
#     'bathrooms',
#     'furnishing',
#     'locality',
#     'society',
#     'floor',
#     'total_floors',
#     'building_age'
# ]

#     main(df1, ml_model, label_encoders=label_encoders, feature_names=features_with_log)
