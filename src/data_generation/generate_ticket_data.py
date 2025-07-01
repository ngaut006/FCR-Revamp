import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

class TicketDataGenerator:
    def __init__(self, num_tickets=150000, start_date="2023-01-01", end_date="2023-12-31"):
        self.fake = Faker()
        self.num_tickets = num_tickets
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Define constant lists for categorical variables
        self.categories = [
            "Software Installation", "Hardware Issues", "Network Problems",
            "Account Access", "Email Issues", "Application Error",
            "System Performance", "Security Incident", "Password Reset",
            "Data Recovery"
        ]
        
        self.subcategories = {
            "Software Installation": ["Installation Failed", "License Issues", "Compatibility Problems", "Update Error"],
            "Hardware Issues": ["Desktop", "Laptop", "Printer", "Monitor", "Peripherals"],
            "Network Problems": ["Internet Down", "VPN Issues", "Slow Connection", "WiFi Problems"],
            "Account Access": ["Login Failed", "Account Locked", "Permission Issues", "MFA Problems"],
            "Email Issues": ["Cannot Send", "Cannot Receive", "Sync Issues", "Storage Full"],
            "Application Error": ["Crash Report", "Feature Not Working", "Performance Issues", "Data Error"],
            "System Performance": ["Slow System", "High CPU Usage", "Memory Issues", "Disk Space"],
            "Security Incident": ["Suspicious Email", "Malware Alert", "Unauthorized Access", "Policy Violation"],
            "Password Reset": ["Forgotten Password", "Expired Password", "Complex Requirements", "Reset Failed"],
            "Data Recovery": ["File Deletion", "Corruption", "Backup Restore", "Transfer Issues"]
        }
        
        self.priorities = ["Low", "Medium", "High", "Critical"]
        self.channels = ["Email", "Phone", "Chat", "Web Portal"]
        self.teams = ["Tier 1", "Tier 2", "Tier 3"]
        
        # Generate agent data
        self.agents = self._generate_agents()

    def _generate_agents(self, num_agents=50):
        agents = []
        for _ in range(num_agents):
            agent = {
                'agent_id': f"AG{str(_+1).zfill(3)}",
                'name': self.fake.name(),
                'team': random.choice(self.teams),
                'experience_years': random.uniform(0.5, 8.0),
                'avg_handling_time': random.uniform(10, 45)  # minutes
            }
            agents.append(agent)
        return agents

    def _generate_ticket_description(self, category, subcategory):
        templates = {
            "Software Installation": "User reported issues with {subcategory}. {detail}",
            "Hardware Issues": "Customer experiencing problems with {subcategory}. {detail}",
            "Network Problems": "User cannot {subcategory}. {detail}",
            "Account Access": "User reported {subcategory}. {detail}",
            "Email Issues": "Email problem: {subcategory}. {detail}",
            "Application Error": "Application issue: {subcategory}. {detail}",
            "System Performance": "System issue: {subcategory}. {detail}",
            "Security Incident": "Security alert: {subcategory}. {detail}",
            "Password Reset": "Password issue: {subcategory}. {detail}",
            "Data Recovery": "Data issue: {subcategory}. {detail}"
        }
        
        details = [
            "User needs immediate assistance.",
            "Impact on business operations.",
            "Requesting urgent support.",
            "Needs resolution within SLA.",
            "Multiple users affected.",
            "Recurring issue reported.",
            "Business critical system affected.",
            "User requesting escalation.",
            "Previous solution didn't work.",
            "Following standard protocol."
        ]
        
        template = templates.get(category, "Issue reported: {subcategory}. {detail}")
        return template.format(subcategory=subcategory, detail=random.choice(details))

    def generate_data(self):
        data = []
        
        for i in range(self.num_tickets):
            # Generate timestamp within date range
            created_at = self.fake.date_time_between(
                start_date=self.start_date,
                end_date=self.end_date
            )
            
            # Select category and subcategory
            category = random.choice(self.categories)
            subcategory = random.choice(self.subcategories[category])
            
            # Select agent
            agent = random.choice(self.agents)
            
            # Generate priority based on category and subcategory
            priority_weights = {
                "Critical": 0.1,
                "High": 0.25,
                "Medium": 0.4,
                "Low": 0.25
            }
            priority = random.choices(
                self.priorities,
                weights=[priority_weights[p] for p in self.priorities]
            )[0]
            
            # Calculate resolution time (in hours)
            base_resolution_time = {
                "Critical": random.uniform(0.5, 4),
                "High": random.uniform(2, 8),
                "Medium": random.uniform(4, 24),
                "Low": random.uniform(12, 48)
            }[priority]
            
            resolution_time = base_resolution_time * (1 + random.uniform(-0.2, 0.2))
            
            # Determine FCR based on various factors
            fcr_probability = 0.8  # base probability
            
            # Adjust FCR probability based on factors
            if priority in ["Critical", "High"]:
                fcr_probability *= 0.9
            if agent['experience_years'] > 5:
                fcr_probability *= 1.1
            if resolution_time > 24:
                fcr_probability *= 0.8
            
            fcr = random.random() < min(fcr_probability, 1.0)
            
            # Generate ticket data
            ticket = {
                'ticket_id': f"TKT{str(i+1).zfill(6)}",
                'created_at': created_at,
                'category': category,
                'subcategory': subcategory,
                'priority': priority,
                'channel': random.choice(self.channels),
                'description': self._generate_ticket_description(category, subcategory),
                'agent_id': agent['agent_id'],
                'agent_name': agent['name'],
                'agent_team': agent['team'],
                'agent_experience': agent['experience_years'],
                'resolution_time': resolution_time,
                'fcr': fcr,
                'customer_satisfaction': random.randint(1, 5) if fcr else random.randint(1, 3)
            }
            
            data.append(ticket)
        
        return pd.DataFrame(data)

    def save_data(self, filename='support_tickets.csv'):
        df = self.generate_data()
        df.to_csv(filename, index=False)
        print(f"Generated {len(df)} tickets and saved to {filename}")
        return df

if __name__ == "__main__":
    generator = TicketDataGenerator()
    df = generator.save_data() 